package com.example.audit;

import openllet.owlapi.OpenlletReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLDataFactory;
import org.semanticweb.owlapi.model.OWLNamedIndividual;
import org.semanticweb.owlapi.model.OWLObjectProperty;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import org.semanticweb.owlapi.reasoner.OWLReasonerFactory;
import org.semanticweb.owlapi.reasoner.structural.StructuralReasonerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Offline batch entailment audit for the post-publication correction.
 *
 * Input is a tab-separated file with columns:
 * task_id, context_path, answer_type, subject_iri, predicate_iri, object_iri.
 * Context rows must be contiguous. Output never mutates benchmark artifacts.
 */
public final class ReasonerBatchAudit {
    private static final String RDF_TYPE =
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

    private ReasonerBatchAudit() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            throw new IllegalArgumentException(
                    "Usage: ReasonerBatchAudit <openllet|structural> <input.tsv> <output.tsv>");
        }
        String mode = args[0].toLowerCase(Locale.ROOT);
        OWLReasonerFactory factory = switch (mode) {
            case "openllet" -> OpenlletReasonerFactory.getInstance();
            case "structural" -> new StructuralReasonerFactory();
            default -> throw new IllegalArgumentException("Unknown reasoner: " + args[0]);
        };

        try (BufferedReader reader = Files.newBufferedReader(Path.of(args[1]), StandardCharsets.UTF_8);
             BufferedWriter writer = Files.newBufferedWriter(Path.of(args[2]), StandardCharsets.UTF_8)) {
            writer.write("task_id\treasoner\tstatus\tconsistent\tentailed\tanswer_iris\terror\n");
            String header = reader.readLine();
            if (header == null) {
                return;
            }
            String currentPath = null;
            List<Query> queries = new ArrayList<>();
            for (String line; (line = reader.readLine()) != null; ) {
                if (line.isBlank()) {
                    continue;
                }
                Query query = Query.parse(line);
                if (currentPath != null && !currentPath.equals(query.contextPath())) {
                    auditContext(mode, factory, currentPath, queries, writer);
                    queries.clear();
                }
                currentPath = query.contextPath();
                queries.add(query);
            }
            if (currentPath != null) {
                auditContext(mode, factory, currentPath, queries, writer);
            }
        }
    }

    private static void auditContext(
            String mode,
            OWLReasonerFactory factory,
            String contextPath,
            List<Query> queries,
            BufferedWriter writer) throws Exception {
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLReasoner reasoner = null;
        try {
            OWLOntology ontology = manager.loadOntologyFromOntologyDocument(Path.of(contextPath).toFile());
            reasoner = factory.createReasoner(ontology);
            boolean consistent = reasoner.isConsistent();
            reasoner.precomputeInferences();
            OWLDataFactory dataFactory = manager.getOWLDataFactory();
            for (Query query : queries) {
                try {
                    List<String> answers = answers(reasoner, dataFactory, query);
                    boolean entailed = query.answerType().equals("BIN") &&
                            answers.contains(query.objectIri());
                    write(writer, query.taskId(), mode, "ok", consistent,
                            entailed, String.join(";", answers), "");
                } catch (Exception exception) {
                    write(writer, query.taskId(), mode, "query_error", consistent,
                            false, "", exception.toString());
                }
            }
        } catch (Exception exception) {
            for (Query query : queries) {
                write(writer, query.taskId(), mode, "context_error", false,
                        false, "", exception.toString());
            }
        } finally {
            if (reasoner != null) {
                reasoner.dispose();
            }
            manager.clearOntologies();
        }
        writer.flush();
    }

    private static List<String> answers(
            OWLReasoner reasoner, OWLDataFactory dataFactory, Query query) {
        OWLNamedIndividual subject = dataFactory.getOWLNamedIndividual(IRI.create(query.subjectIri()));
        List<String> result;
        if (query.predicateIri().equals(RDF_TYPE)) {
            result = reasoner.getTypes(subject, false).entities()
                    .filter(value -> !value.isOWLThing() && !value.isOWLNothing())
                    .map(value -> value.getIRI().toString())
                    .collect(Collectors.toCollection(ArrayList::new));
        } else {
            OWLObjectProperty property = dataFactory.getOWLObjectProperty(IRI.create(query.predicateIri()));
            result = reasoner.getObjectPropertyValues(subject, property).entities()
                    .map(value -> value.getIRI().toString())
                    .collect(Collectors.toCollection(ArrayList::new));
        }
        result.sort(Comparator.naturalOrder());
        return result.stream().distinct().collect(Collectors.toList());
    }

    private static void write(
            BufferedWriter writer, String taskId, String reasoner, String status,
            boolean consistent, boolean entailed, String answers, String error) throws Exception {
        writer.write(String.join("\t",
                clean(taskId), clean(reasoner), clean(status), Boolean.toString(consistent),
                Boolean.toString(entailed), clean(answers), clean(error)));
        writer.newLine();
    }

    private static String clean(String value) {
        return String.valueOf(value == null ? "" : value)
                .replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
    }

    private record Query(
            String taskId, String contextPath, String answerType,
            String subjectIri, String predicateIri, String objectIri) {
        private static Query parse(String line) {
            String[] values = line.split("\t", -1);
            if (values.length != 6) {
                throw new IllegalArgumentException("Expected 6 TSV fields, got " + values.length);
            }
            return new Query(values[0], values[1], values[2], values[3], values[4], values[5]);
        }
    }
}
