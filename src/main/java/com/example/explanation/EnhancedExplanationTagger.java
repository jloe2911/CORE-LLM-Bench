package com.example.explanation;

import org.semanticweb.owlapi.model.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Canonical 20-tag classifier with recursive OWLAPI class-expression traversal. */
public class EnhancedExplanationTagger {
    public static final String TAG_DIRECT = "D";
    public static final String TAG_HIERARCHY = "H";
    public static final String TAG_TRANSITIVITY = "T";
    public static final String TAG_SYMMETRY = "S";
    public static final String TAG_ASYMMETRIC = "A";
    public static final String TAG_DISJOINT = "J";
    public static final String TAG_CHAIN = "N";
    public static final String TAG_EXISTENTIAL = "E";
    public static final String TAG_INTERSECTION = "∩";
    public static final String TAG_COMPLEMENT = "¬";
    public static final String TAG_INVERSE = "I";
    public static final String TAG_FUNCTIONAL = "F";
    public static final String TAG_REFLEXIVE = "V";
    public static final String TAG_IRREFLEXIVE = "Y";
    public static final String TAG_EQUIVALENCE = "Q";
    public static final String TAG_DOMAIN_RANGE = "R";
    public static final String TAG_CARDINALITY = "C";
    public static final String TAG_UNIVERSAL = "L";
    public static final String TAG_UNION = "U";
    public static final String TAG_MULTI_STEP = "M";

    private static final List<String> CANONICAL_ORDER = Arrays.asList(
            TAG_DIRECT, TAG_HIERARCHY, TAG_TRANSITIVITY, TAG_SYMMETRY,
            TAG_ASYMMETRIC, TAG_DISJOINT, TAG_CHAIN, TAG_EXISTENTIAL,
            TAG_INTERSECTION, TAG_COMPLEMENT, TAG_INVERSE, TAG_FUNCTIONAL,
            TAG_REFLEXIVE, TAG_IRREFLEXIVE, TAG_EQUIVALENCE,
            TAG_DOMAIN_RANGE, TAG_CARDINALITY, TAG_UNIVERSAL, TAG_UNION,
            TAG_MULTI_STEP);

    /** De-duplicates semantic axioms and never re-counts their text rendering. */
    public String tagExplanation(ExplanationPath path) {
        if (path == null) return "";
        List<TagOccurrence> occurrences = new ArrayList<>();
        Set<OWLAxiom> processedAxioms = new LinkedHashSet<>();
        for (OWLAxiom axiom : path.getAxioms()) {
            OWLAxiom semanticAxiom = axiom.getAxiomWithoutAnnotations();
            if (!processedAxioms.add(semanticAxiom)) continue;
            appendOccurrences(occurrences, tagSingleAxiom(semanticAxiom),
                    "axiom:" + semanticAxiom);
        }

        // Justifications are display serializations of structured axioms.  They
        // are a fallback only, preventing historic H/I/N/S/T doubling.
        if (processedAxioms.isEmpty() && path.getJustifications() != null) {
            Set<String> processedText = new LinkedHashSet<>();
            for (String justification : path.getJustifications()) {
                String identity = normalizeText(justification);
                if (identity.isEmpty() || !processedText.add(identity)) continue;
                appendOccurrences(occurrences, tagJustificationString(justification),
                        "text:" + identity);
            }
        }

        Set<String> distinctNonDirectTypes = new LinkedHashSet<>();
        for (TagOccurrence occurrence : occurrences) {
            if (!TAG_DIRECT.equals(occurrence.tag())) {
                distinctNonDirectTypes.add(occurrence.tag());
            }
        }
        if (distinctNonDirectTypes.size() >= 2) {
            occurrences.add(new TagOccurrence(TAG_MULTI_STEP, "meta:heterogeneous-tbox"));
        }
        return buildOrderedTagString(occurrences);
    }

    /** M is categorical metadata and is excluded from primitive complexity. */
    public int primitiveTagCount(ExplanationPath path) {
        return (int) tagExplanation(path).codePoints()
                .filter(cp -> cp != TAG_MULTI_STEP.codePointAt(0)).count();
    }

    List<String> tagSingleAxiom(OWLAxiom axiom) {
        List<String> tags = new ArrayList<>();
        if (axiom == null) return tags;
        if (axiom instanceof OWLClassAssertionAxiom classAssertion) {
            tags.add(TAG_DIRECT);
            collectClassExpressionTags(classAssertion.getClassExpression(), tags);
        } else if (axiom instanceof OWLObjectPropertyAssertionAxiom
                || axiom instanceof OWLDataPropertyAssertionAxiom) {
            tags.add(TAG_DIRECT);
        } else if (axiom instanceof OWLSubClassOfAxiom subClass) {
            tags.add(TAG_HIERARCHY);
            collectClassExpressionTags(subClass.getSubClass(), tags);
            collectClassExpressionTags(subClass.getSuperClass(), tags);
        } else if (axiom instanceof OWLSubObjectPropertyOfAxiom
                || axiom instanceof OWLSubDataPropertyOfAxiom) {
            tags.add(TAG_HIERARCHY);
        } else if (axiom instanceof OWLEquivalentClassesAxiom equivalent) {
            tags.add(TAG_EQUIVALENCE);
            equivalent.getClassExpressions().forEach(e -> collectClassExpressionTags(e, tags));
        } else if (axiom instanceof OWLEquivalentObjectPropertiesAxiom
                || axiom instanceof OWLEquivalentDataPropertiesAxiom) {
            tags.add(TAG_EQUIVALENCE);
        } else if (axiom instanceof OWLDisjointClassesAxiom disjoint) {
            tags.add(TAG_DISJOINT);
            disjoint.getClassExpressions().forEach(e -> collectClassExpressionTags(e, tags));
        } else if (axiom instanceof OWLDisjointObjectPropertiesAxiom
                || axiom instanceof OWLDisjointDataPropertiesAxiom) {
            tags.add(TAG_DISJOINT);
        } else if (axiom instanceof OWLSubPropertyChainOfAxiom) tags.add(TAG_CHAIN);
        else if (axiom instanceof OWLTransitiveObjectPropertyAxiom) tags.add(TAG_TRANSITIVITY);
        else if (axiom instanceof OWLSymmetricObjectPropertyAxiom) tags.add(TAG_SYMMETRY);
        else if (axiom instanceof OWLAsymmetricObjectPropertyAxiom) tags.add(TAG_ASYMMETRIC);
        else if (axiom instanceof OWLReflexiveObjectPropertyAxiom) tags.add(TAG_REFLEXIVE);
        else if (axiom instanceof OWLIrreflexiveObjectPropertyAxiom) tags.add(TAG_IRREFLEXIVE);
        else if (axiom instanceof OWLInverseObjectPropertiesAxiom) tags.add(TAG_INVERSE);
        else if (axiom instanceof OWLFunctionalObjectPropertyAxiom
                || axiom instanceof OWLFunctionalDataPropertyAxiom
                || axiom instanceof OWLInverseFunctionalObjectPropertyAxiom) tags.add(TAG_FUNCTIONAL);
        else if (axiom instanceof OWLObjectPropertyDomainAxiom domain) {
            tags.add(TAG_DOMAIN_RANGE); collectClassExpressionTags(domain.getDomain(), tags);
        } else if (axiom instanceof OWLObjectPropertyRangeAxiom range) {
            tags.add(TAG_DOMAIN_RANGE); collectClassExpressionTags(range.getRange(), tags);
        } else if (axiom instanceof OWLDataPropertyDomainAxiom domain) {
            tags.add(TAG_DOMAIN_RANGE); collectClassExpressionTags(domain.getDomain(), tags);
        } else if (axiom instanceof OWLDataPropertyRangeAxiom) tags.add(TAG_DOMAIN_RANGE);
        return tags;
    }

    private void collectClassExpressionTags(OWLClassExpression expression, List<String> tags) {
        if (expression == null || !expression.isAnonymous()) return;
        if (expression instanceof OWLObjectIntersectionOf x) {
            tags.add(TAG_INTERSECTION); x.getOperands().forEach(e -> collectClassExpressionTags(e, tags));
        } else if (expression instanceof OWLObjectUnionOf x) {
            tags.add(TAG_UNION); x.getOperands().forEach(e -> collectClassExpressionTags(e, tags));
        } else if (expression instanceof OWLObjectComplementOf x) {
            tags.add(TAG_COMPLEMENT); collectClassExpressionTags(x.getOperand(), tags);
        } else if (expression instanceof OWLObjectSomeValuesFrom x) {
            tags.add(TAG_EXISTENTIAL); collectClassExpressionTags(x.getFiller(), tags);
        } else if (expression instanceof OWLDataSomeValuesFrom) tags.add(TAG_EXISTENTIAL);
        else if (expression instanceof OWLObjectAllValuesFrom x) {
            tags.add(TAG_UNIVERSAL); collectClassExpressionTags(x.getFiller(), tags);
        } else if (expression instanceof OWLDataAllValuesFrom) tags.add(TAG_UNIVERSAL);
        else if (expression instanceof OWLObjectMinCardinality x) {
            tags.add(TAG_CARDINALITY); collectClassExpressionTags(x.getFiller(), tags);
        } else if (expression instanceof OWLObjectMaxCardinality x) {
            tags.add(TAG_CARDINALITY); collectClassExpressionTags(x.getFiller(), tags);
        } else if (expression instanceof OWLObjectExactCardinality x) {
            tags.add(TAG_CARDINALITY); collectClassExpressionTags(x.getFiller(), tags);
        } else if (expression instanceof OWLDataMinCardinality
                || expression instanceof OWLDataMaxCardinality
                || expression instanceof OWLDataExactCardinality) tags.add(TAG_CARDINALITY);
    }

    private List<String> tagJustificationString(String justification) {
        String lower = normalizeText(justification);
        List<String> tags = new ArrayList<>();
        addIf(tags, TAG_DIRECT, lower.contains("rdf:type") || lower.contains("classassertion")
                || lower.contains("propertyassertion"));
        addIf(tags, TAG_HIERARCHY, lower.contains("subclassof") || lower.contains("subpropertyof"));
        addIf(tags, TAG_TRANSITIVITY, lower.contains("transitiveobjectproperty"));
        addIf(tags, TAG_ASYMMETRIC, lower.contains("asymmetricobjectproperty"));
        addIf(tags, TAG_SYMMETRY, lower.contains("symmetricobjectproperty")
                && !lower.contains("asymmetricobjectproperty"));
        addIf(tags, TAG_DISJOINT, lower.contains("disjoint"));
        addIf(tags, TAG_CHAIN, lower.contains("propertychain"));
        addIf(tags, TAG_EXISTENTIAL, lower.contains("somevaluesfrom"));
        addIf(tags, TAG_INTERSECTION, lower.contains("intersectionof"));
        addIf(tags, TAG_COMPLEMENT, lower.contains("complementof"));
        addIf(tags, TAG_INVERSE, lower.contains("inverseof"));
        addIf(tags, TAG_FUNCTIONAL, lower.contains("functional"));
        addIf(tags, TAG_REFLEXIVE, lower.contains("reflexiveobjectproperty")
                && !lower.contains("irreflexiveobjectproperty"));
        addIf(tags, TAG_IRREFLEXIVE, lower.contains("irreflexiveobjectproperty"));
        addIf(tags, TAG_EQUIVALENCE, lower.contains("equivalentclass")
                || lower.contains("equivalentpropert"));
        addIf(tags, TAG_DOMAIN_RANGE, lower.contains("domain(") || lower.contains("range(")
                || lower.contains("rdfs:domain") || lower.contains("rdfs:range"));
        addIf(tags, TAG_CARDINALITY, lower.contains("cardinality"));
        addIf(tags, TAG_UNIVERSAL, lower.contains("allvaluesfrom"));
        addIf(tags, TAG_UNION, lower.contains("unionof"));
        return tags;
    }

    private static void addIf(Collection<String> tags, String tag, boolean condition) {
        if (condition) tags.add(tag);
    }

    private static void appendOccurrences(List<TagOccurrence> target, List<String> tags, String source) {
        for (int i = 0; i < tags.size(); i++) target.add(new TagOccurrence(tags.get(i), source + "#" + i));
    }

    private String buildOrderedTagString(List<TagOccurrence> occurrences) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        Set<String> identities = new LinkedHashSet<>();
        for (TagOccurrence occurrence : occurrences) {
            if (identities.add(occurrence.sourceIdentity())) counts.merge(occurrence.tag(), 1, Integer::sum);
        }
        StringBuilder result = new StringBuilder();
        for (String tag : CANONICAL_ORDER) result.append(tag.repeat(counts.getOrDefault(tag, 0)));
        return result.toString();
    }

    private static String normalizeText(String value) {
        return value == null ? "" : value.trim().replaceAll("\\s+", " ").toLowerCase(Locale.ROOT);
    }

    private record TagOccurrence(String tag, String sourceIdentity) {}
}
