package com.example.explanation;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;

class EnhancedExplanationTaggerTest {
    private OWLDataFactory df;
    private EnhancedExplanationTagger tagger;
    private OWLClass a, b, c;
    private OWLObjectProperty p, q;

    @BeforeEach void setUp() {
        df = OWLManager.getOWLDataFactory(); tagger = new EnhancedExplanationTagger();
        a = cls("A"); b = cls("B"); c = cls("C"); p = prop("p"); q = prop("q");
    }

    @Test void nestedClassExpressionsContributeMultipleTags() {
        assertEquals("H", tag(df.getOWLSubClassOfAxiom(a, b)));
        assertEquals("HEM", tag(df.getOWLSubClassOfAxiom(a, df.getOWLObjectSomeValuesFrom(p, b))));
        assertEquals("E∩QM", tag(df.getOWLEquivalentClassesAxiom(a,
                df.getOWLObjectIntersectionOf(b, df.getOWLObjectSomeValuesFrom(p, c)))));
        assertEquals("∩QCM", tag(df.getOWLEquivalentClassesAxiom(a,
                df.getOWLObjectIntersectionOf(b, df.getOWLObjectMinCardinality(3, p, c)))));
        assertEquals("HLM", tag(df.getOWLSubClassOfAxiom(a, df.getOWLObjectAllValuesFrom(p, b))));
        assertEquals("QUM", tag(df.getOWLEquivalentClassesAxiom(a, df.getOWLObjectUnionOf(b, c))));
        assertEquals("¬QM", tag(df.getOWLEquivalentClassesAxiom(a, df.getOWLObjectComplementOf(b))));
    }

    @Test void coversCanonicalPropertyAndTboxTags() {
        assertEquals("N", tag(df.getOWLSubPropertyChainOfAxiom(List.of(p, q), p)));
        assertEquals("I", tag(df.getOWLInverseObjectPropertiesAxiom(p, q)));
        assertEquals("S", tag(df.getOWLSymmetricObjectPropertyAxiom(p)));
        assertEquals("A", tag(df.getOWLAsymmetricObjectPropertyAxiom(p)));
        assertEquals("T", tag(df.getOWLTransitiveObjectPropertyAxiom(p)));
        assertEquals("F", tag(df.getOWLFunctionalObjectPropertyAxiom(p)));
        assertEquals("V", tag(df.getOWLReflexiveObjectPropertyAxiom(p)));
        assertEquals("Y", tag(df.getOWLIrreflexiveObjectPropertyAxiom(p)));
        assertEquals("R", tag(df.getOWLObjectPropertyDomainAxiom(p, a)));
        assertEquals("J", tag(df.getOWLDisjointClassesAxiom(a, b)));
    }

    @Test void duplicateAxiomAndTextRenderingAreNotDoubleCounted() {
        OWLAxiom hierarchy = df.getOWLSubClassOfAxiom(a, b);
        ExplanationPath path = path(List.of(hierarchy, hierarchy));
        path.setJustifications(List.of(hierarchy.toString(), hierarchy.toString()));
        assertEquals("H", tagger.tagExplanation(path));
    }

    @Test void mMeansDistinctNonDirectTypesAndIsNotPrimitive() {
        OWLAxiom direct = df.getOWLClassAssertionAxiom(a, df.getOWLNamedIndividual(iri("x")));
        OWLAxiom h1 = df.getOWLSubClassOfAxiom(a, b), h2 = df.getOWLSubClassOfAxiom(b, c);
        assertFalse(tag(path(List.of(direct))).contains("M"));
        assertFalse(tag(path(List.of(direct, h1, h2))).contains("M"));
        ExplanationPath heterogeneous = path(List.of(direct, h1, df.getOWLObjectPropertyDomainAxiom(p, a)));
        assertTrue(tag(heterogeneous).contains("M"));
        assertEquals(3, tagger.primitiveTagCount(heterogeneous));
    }

    @Test void structuredFormatterDeduplicatesSemanticAxiomsAndSeparatesM() {
        OWLAxiom direct = df.getOWLClassAssertionAxiom(a, df.getOWLNamedIndividual(iri("x")));
        OWLAxiom nested = df.getOWLSubClassOfAxiom(
                a,
                df.getOWLObjectSomeValuesFrom(
                        p,
                        df.getOWLObjectIntersectionOf(
                                b, df.getOWLObjectMinCardinality(2, q, c))));
        ExplanationPath explanation = path(List.of(direct, direct, nested));
        String json = ExplanationFormatter.generateExactJSONFormat(
                "x|rdf:type|A", Set.of(explanation), tagger);

        assertTrue(json.contains("\"structuredExplanations\""));
        assertTrue(json.contains("\"axiomCount\" : 2"));
        assertTrue(json.contains("\"primitiveTagCount\" : 5"));
        assertTrue(json.contains("\"m\" : true"));
        assertEquals(1, countOccurrences(
                json,
                "\"identity\" : \"" + direct.getAxiomWithoutAnnotations()));
    }

    private String tag(OWLAxiom axiom) { return tag(path(List.of(axiom))); }
    private String tag(ExplanationPath path) { return tagger.tagExplanation(path); }
    private ExplanationPath path(List<OWLAxiom> axioms) {
        return new ExplanationPath(axioms, "test", ExplanationType.DIRECT_ASSERTION, axioms.size());
    }
    private OWLClass cls(String name) { return df.getOWLClass(iri(name)); }
    private OWLObjectProperty prop(String name) { return df.getOWLObjectProperty(iri(name)); }
    private IRI iri(String name) { return IRI.create("urn:test:" + name); }
    private int countOccurrences(String value, String needle) {
        return (value.length() - value.replace(needle, "").length()) / needle.length();
    }
}
