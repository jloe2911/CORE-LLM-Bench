// com/example/explanation/ExplanationFormatter.java
package com.example.explanation;

import com.example.processing.GlobalQueryTracker;
import com.example.util.OntologyUtils;
import com.example.util.URIUtils;
import org.semanticweb.owlapi.model.OWLAxiom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Handles formatting of explanations into various output formats
 */
public class ExplanationFormatter {

    private static final Logger LOGGER = LoggerFactory.getLogger(ExplanationFormatter.class);

    /**
     * Generate the exact JSON format for explanations with proper task ID references and BOTH SPARQL queries
     */
    public static String generateExactJSONFormat(String tripleKey, Set<ExplanationPath> paths,
                                                 EnhancedExplanationTagger tagger) {
        return generateExactJSONFormat(tripleKey, tripleKey, paths, tagger);
    }

    /**
     * Generate the exact JSON format using a stable explanation key while keeping
     * the inferred triple in subject|predicate|object format.
     */
    public static String generateExactJSONFormat(String explanationKey, String tripleKey, Set<ExplanationPath> paths,
                                                 EnhancedExplanationTagger tagger) {
        StringBuilder json = new StringBuilder();
        String[] parts = OntologyUtils.parseTripleKey(tripleKey);

        if (parts.length != 3) {
            LOGGER.warn("Invalid triple key format: {}", tripleKey);
            return "";
        }

        json.append("  \"").append(OntologyUtils.escapeJson(explanationKey)).append("\" : {\n");

        // Inferred section uses short forms
        json.append("    \"inferred\" : {\n");
        json.append("      \"subject\" : \"").append(parts[0]).append("\",\n");
        json.append("      \"predicate\" : \"").append(parts[1]).append("\",\n");
        json.append("      \"object\" : \"").append(parts[2]).append("\"\n");
        json.append("    },\n");

        // Explanations section
        json.append("    \"explanations\" : [ ");
        List<ExplanationPath> pathList = paths.stream()
                .sorted(Comparator.comparing(ExplanationFormatter::semanticPathIdentity))
                .collect(Collectors.toList());
        for (int i = 0; i < pathList.size(); i++) {
            ExplanationPath path = pathList.get(i);

            json.append("[ ");

            // Add all justification axioms for this path
            List<String> justifications = extractJustificationsFromPath(path);
            for (int j = 0; j < justifications.size(); j++) {
                json.append("\"").append(OntologyUtils.escapeJson(justifications.get(j))).append("\"");
                if (j < justifications.size() - 1) {
                    json.append(", ");
                }
            }

            // Add the tag for this specific path
            String pathTag = tagger.tagExplanation(path);
            if (!pathTag.isEmpty()) {
                json.append(", \"TAG:").append(pathTag).append("\"");
            }

            json.append(" ]");
            if (i < pathList.size() - 1) {
                json.append(", ");
            }
        }
        json.append(" ],\n");

        // Machine-readable semantic provenance for v1.1 complexity auditing.
        // This remains separate from the legacy display-oriented explanations.
        json.append("    \"structuredExplanations\" : [ ");
        for (int i = 0; i < pathList.size(); i++) {
            appendStructuredExplanation(json, pathList.get(i), tagger);
            if (i < pathList.size() - 1) {
                json.append(", ");
            }
        }
        json.append(" ],\n");

        // Size section
        if (!paths.isEmpty()) {
            int minSize = paths.stream().mapToInt(ExplanationPath::getComplexity).min().orElse(1);
            int maxSize = paths.stream().mapToInt(ExplanationPath::getComplexity).max().orElse(1);

            json.append("    \"size\" : {\n");
            json.append("      \"min\" : ").append(minSize).append(",\n");
            json.append("      \"max\" : ").append(maxSize).append("\n");
            json.append("    },\n");
        }

        json.append("    \"explanationCount\" : ").append(paths.size()).append(",\n");

        // Task IDs section - get ALL task IDs for this triple (both BIN and MC)
        List<String> allTaskIds = GlobalQueryTracker.getTaskIds(explanationKey);
        json.append("    \"taskIds\" : [ ");
        for (int i = 0; i < allTaskIds.size(); i++) {
            json.append("\"").append(allTaskIds.get(i)).append("\"");
            if (i < allTaskIds.size() - 1) {
                json.append(", ");
            }
        }
        json.append(" ],\n");

        // SPARQL queries section - include BOTH ASK and SELECT queries
        json.append("    \"sparqlQueries\" : [ ");

        // Always include the ASK query
        json.append("\"ASK WHERE { <").append(URIUtils.getFullURI(parts[0]))
                .append("> <").append(URIUtils.getFullURI(parts[1])).append("> <")
                .append(URIUtils.getFullURI(parts[2])).append("> }\"");

        // Add SELECT query if MC task exists
        boolean hasMCTask = allTaskIds.stream().anyMatch(taskId -> taskId.contains("-MC"));
        if (hasMCTask) {
            json.append(", \"SELECT ?x WHERE { <").append(URIUtils.getFullURI(parts[0]))
                    .append("> <").append(URIUtils.getFullURI(parts[1])).append("> ?x }\"");
        }

        json.append(" ]\n");

        json.append("  }");
        return json.toString();
    }

    /**
     * Extract justifications from an explanation path
     */
    public static List<String> extractJustificationsFromPath(ExplanationPath path) {
        List<String> justifications = new ArrayList<>();

        if (path == null) {
            return justifications;
        }

        // Priority 1: Use path justifications if available and not empty
        if (path.getJustifications() != null && !path.getJustifications().isEmpty()) {
            return new ArrayList<>(path.getJustifications());
        }

        // Priority 2: Convert axioms to justifications using standardized formatting
        if (path.getAxioms() != null && !path.getAxioms().isEmpty()) {
            for (OWLAxiom axiom : path.getAxioms()) {
                String formatted = OntologyUtils.formatAxiom(axiom);
                if (formatted != null && !formatted.trim().isEmpty()) {
                    justifications.add(formatted);
                }
            }
        }

        // Priority 3: Fallback to description
        if (justifications.isEmpty() && path.getDescription() != null && !path.getDescription().trim().isEmpty()) {
            justifications.add(path.getDescription());
        }

        // Ensure we always have at least one justification
        if (justifications.isEmpty()) {
            justifications.add("Explanation available but details unclear");
        }

        return justifications;
    }

    private static List<OWLAxiom> semanticAxioms(ExplanationPath path) {
        if (path == null || path.getAxioms() == null) {
            return List.of();
        }
        Set<OWLAxiom> unique = new LinkedHashSet<>();
        for (OWLAxiom axiom : path.getAxioms()) {
            unique.add(axiom.getAxiomWithoutAnnotations());
        }
        return unique.stream()
                .sorted(Comparator.comparing(OWLAxiom::toString))
                .collect(Collectors.toList());
    }

    private static String semanticPathIdentity(ExplanationPath path) {
        return semanticAxioms(path).stream()
                .map(OWLAxiom::toString)
                .collect(Collectors.joining("\n"));
    }

    private static void appendStructuredExplanation(
            StringBuilder json, ExplanationPath path, EnhancedExplanationTagger tagger) {
        List<OWLAxiom> axioms = semanticAxioms(path);
        String tagString = tagger.tagExplanation(path);
        List<String> primitiveTags = tagString.codePoints()
                .mapToObj(cp -> new String(Character.toChars(cp)))
                .filter(tag -> !EnhancedExplanationTagger.TAG_MULTI_STEP.equals(tag))
                .collect(Collectors.toList());
        Set<String> distinctTags = new LinkedHashSet<>(primitiveTags);

        json.append("{ \"semanticAxioms\" : [ ");
        for (int i = 0; i < axioms.size(); i++) {
            OWLAxiom axiom = axioms.get(i);
            List<String> axiomTags = tagger.tagSingleAxiom(axiom);
            json.append("{ \"identity\" : \"")
                    .append(OntologyUtils.escapeJson(axiom.toString()))
                    .append("\", \"rendering\" : \"")
                    .append(OntologyUtils.escapeJson(OntologyUtils.formatAxiom(axiom)))
                    .append("\", \"primitiveTags\" : ");
            appendStringArray(json, axiomTags);
            json.append(" }");
            if (i < axioms.size() - 1) json.append(", ");
        }
        json.append(" ], \"primitiveTags\" : ");
        appendStringArray(json, primitiveTags);
        json.append(", \"distinctPrimitiveTags\" : ");
        appendStringArray(json, new ArrayList<>(distinctTags));
        json.append(", \"axiomCount\" : ").append(axioms.size())
                .append(", \"primitiveTagCount\" : ").append(primitiveTags.size())
                .append(", \"m\" : ")
                .append(tagString.contains(EnhancedExplanationTagger.TAG_MULTI_STEP))
                .append(" }");
    }

    private static void appendStringArray(StringBuilder json, List<String> values) {
        json.append("[ ");
        for (int i = 0; i < values.size(); i++) {
            json.append("\"").append(OntologyUtils.escapeJson(values.get(i))).append("\"");
            if (i < values.size() - 1) json.append(", ");
        }
        json.append(" ]");
    }

}
