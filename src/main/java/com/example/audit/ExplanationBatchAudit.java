package com.example.audit;

import com.example.explanation.EnhancedExplanationTagger;
import com.example.explanation.ExplanationPath;
import com.example.explanation.ExplanationType;
import com.example.explanation.SemanticAxiomRenderer;
import com.fasterxml.jackson.databind.ObjectMapper;
import openllet.owlapi.OpenlletReasoner;
import openllet.owlapi.OpenlletReasonerFactory;
import openllet.owlapi.explanation.PelletExplanation;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLAxiom;
import org.semanticweb.owlapi.model.OWLNamedIndividual;
import org.semanticweb.owlapi.model.OWLObjectProperty;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Generate deterministic symbolic proof metadata only for newly recovered gold entailments. */
public final class ExplanationBatchAudit {
    private static final String RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    private ExplanationBatchAudit() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: ExplanationBatchAudit <input.tsv> <output.jsonl>");
        }
        ObjectMapper mapper = new ObjectMapper();
        EnhancedExplanationTagger tagger = new EnhancedExplanationTagger();
        PelletExplanation.setup();
        try (BufferedReader input = Files.newBufferedReader(Path.of(args[0]), StandardCharsets.UTF_8);
             BufferedWriter output = Files.newBufferedWriter(Path.of(args[1]), StandardCharsets.UTF_8)) {
            input.readLine();
            String currentContext = null;
            List<Query> queries = new ArrayList<>();
            for (String line; (line = input.readLine()) != null; ) {
                if (line.isBlank()) continue;
                Query query = Query.parse(line);
                if (currentContext != null && !currentContext.equals(query.contextPath())) {
                    process(currentContext, queries, output, mapper, tagger);
                    queries.clear();
                }
                currentContext = query.contextPath();
                queries.add(query);
            }
            if (currentContext != null) process(currentContext, queries, output, mapper, tagger);
        }
    }

    private static void process(String context, List<Query> queries, BufferedWriter output,
                                ObjectMapper mapper, EnhancedExplanationTagger tagger) throws Exception {
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(Path.of(context).toFile());
        OpenlletReasoner reasoner = OpenlletReasonerFactory.getInstance().createReasoner(ontology);
        reasoner.precomputeInferences();
        PelletExplanation explanation = new PelletExplanation(reasoner);
        OWLDataFactory factory = manager.getOWLDataFactory();
        for (Query query : queries) {
            OWLNamedIndividual subject = factory.getOWLNamedIndividual(IRI.create(query.subject()));
            OWLAxiom entailment;
            if (RDF_TYPE.equals(query.predicate())) {
                entailment = factory.getOWLClassAssertionAxiom(
                        factory.getOWLClass(IRI.create(query.object())), subject);
            } else {
                OWLObjectProperty property = factory.getOWLObjectProperty(IRI.create(query.predicate()));
                OWLNamedIndividual object = factory.getOWLNamedIndividual(IRI.create(query.object()));
                entailment = factory.getOWLObjectPropertyAssertionAxiom(property, subject, object);
            }
            Set<OWLAxiom> minimumJustification = explanation.getEntailmentExplanation(entailment);
            Set<Set<OWLAxiom>> justifications = Set.of(minimumJustification);
            List<ExplanationPath> paths = justifications.stream().map(axioms ->
                    new ExplanationPath(new ArrayList<>(axioms),
                            "Openllet minimal entailment justification",
                            ExplanationType.DIRECT_ASSERTION, axioms.size())).toList();
            List<Map<String, Object>> rendered = new ArrayList<>();
            paths.stream().sorted(Comparator.comparing(ExplanationBatchAudit::identity)).forEach(path -> {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("functional_axiom_identities", path.getAxioms().stream()
                        .map(axiom -> axiom.getAxiomWithoutAnnotations().toString()).sorted().toList());
                item.put("manchester_axioms", path.getAxioms().stream()
                        .map(SemanticAxiomRenderer::render).sorted().toList());
                item.put("reasoning_tag", tagger.tagExplanation(path));
                item.put("primitive_tag_count", tagger.primitiveTagCount(path));
                rendered.add(item);
            });
            Map<String, Object> record = new LinkedHashMap<>();
            record.put("task_id", query.taskId());
            record.put("subject_iri", query.subject());
            record.put("predicate_iri", query.predicate());
            record.put("object_iri", query.object());
            record.put("proof_count", rendered.size());
            record.put("proofs", rendered);
            output.write(mapper.writeValueAsString(record));
            output.newLine();
            output.flush();
        }
        reasoner.dispose();
        manager.clearOntologies();
    }

    private static String identity(ExplanationPath path) {
        return path.getAxioms().stream().map(axiom -> axiom.getAxiomWithoutAnnotations().toString())
                .sorted().reduce("", (left, right) -> left + "\n" + right);
    }

    private record Query(String taskId, String contextPath, String subject, String predicate, String object) {
        private static Query parse(String line) {
            String[] values = line.split("\t", -1);
            if (values.length != 5) throw new IllegalArgumentException("Expected five TSV fields");
            return new Query(values[0], values[1], values[2], values[3], values[4]);
        }
    }
}
