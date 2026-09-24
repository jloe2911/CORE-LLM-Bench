package com.example.explanation;

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.io.StringDocumentSource;
import org.semanticweb.owlapi.manchestersyntax.renderer.ManchesterOWLSyntaxOWLObjectRendererImpl;
import org.semanticweb.owlapi.model.OWLAxiom;
import org.semanticweb.owlapi.model.OWLEntity;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.util.ShortFormProvider;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Deterministic OWLAPI 5.1.20 Manchester renderer for serialized semantic
 * axiom identities. Entity IRIs are always emitted in full inside angle
 * brackets: no ontology prefix map or fragment-shortening policy is used.
 */
public final class SemanticAxiomRenderer {
    public static final String OWLAPI_VERSION = "5.1.20";
    public static final String RENDERER =
            "ManchesterOWLSyntaxOWLObjectRendererImpl";
    public static final String PREFIX_POLICY = "full-iri-no-prefixes";

    private static final ShortFormProvider FULL_IRI_PROVIDER = new ShortFormProvider() {
        @Override
        public String getShortForm(OWLEntity entity) {
            return "<" + entity.getIRI() + ">";
        }

        @Override
        public void dispose() {
            // No resources are held.
        }
    };

    private SemanticAxiomRenderer() {
    }

    public static String render(OWLAxiom axiom) {
        ManchesterOWLSyntaxOWLObjectRendererImpl renderer =
                new ManchesterOWLSyntaxOWLObjectRendererImpl();
        renderer.setShortFormProvider(FULL_IRI_PROVIDER);
        return renderer.render(axiom.getAxiomWithoutAnnotations());
    }

    public static OWLAxiom parseFunctionalIdentity(String identity) throws Exception {
        String document = "Ontology(<urn:core-llm-bench:axiom-render>\n"
                + identity + "\n)";
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(
                new StringDocumentSource(document));
        List<OWLAxiom> axioms = new ArrayList<>();
        for (OWLAxiom axiom : ontology.getAxioms()) {
            axioms.add(axiom.getAxiomWithoutAnnotations());
        }
        axioms.sort(Comparator.comparing(OWLAxiom::toString));
        if (axioms.size() != 1) {
            throw new IllegalArgumentException(
                    "Expected exactly one semantic axiom, found " + axioms.size());
        }
        return axioms.get(0);
    }

    public static Map<String, OWLAxiom> parseFunctionalIdentities(
            List<String> identities) throws Exception {
        String document = "Ontology(<urn:core-llm-bench:axiom-render>\n"
                + String.join("\n", identities) + "\n)";
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(
                new StringDocumentSource(document));
        Map<String, OWLAxiom> parsed = new LinkedHashMap<>();
        for (OWLAxiom axiom : ontology.getAxioms()) {
            OWLAxiom semantic = axiom.getAxiomWithoutAnnotations();
            parsed.put(semantic.toString(), semantic);
        }
        if (!parsed.keySet().containsAll(identities)) {
            throw new IllegalArgumentException(
                    "Batch functional-syntax parse did not preserve every identity");
        }
        return parsed;
    }

    /**
     * Batch protocol: input and output are UTF-8 TSV with base64(identity) and
     * base64(rendering). Base64 avoids escaping any OWL or Manchester syntax.
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: SemanticAxiomRenderer INPUT OUTPUT");
        }
        Base64.Decoder decoder = Base64.getDecoder();
        Base64.Encoder encoder = Base64.getEncoder();
        List<String> identities = new ArrayList<>();
        try (BufferedReader input = Files.newBufferedReader(
                Path.of(args[0]), StandardCharsets.UTF_8)) {
            String line;
            while ((line = input.readLine()) != null) {
                if (line.isBlank()) continue;
                identities.add(new String(decoder.decode(line), StandardCharsets.UTF_8));
            }
        }
        Map<String, OWLAxiom> parsed = parseFunctionalIdentities(identities);
        try (BufferedWriter output = Files.newBufferedWriter(
                Path.of(args[1]), StandardCharsets.UTF_8)) {
            for (String identity : identities) {
                String encodedIdentity = encoder.encodeToString(
                        identity.getBytes(StandardCharsets.UTF_8));
                String rendering = render(parsed.get(identity));
                output.write(encodedIdentity);
                output.write('\t');
                output.write(encoder.encodeToString(rendering.getBytes(StandardCharsets.UTF_8)));
                output.newLine();
            }
        }
    }
}
