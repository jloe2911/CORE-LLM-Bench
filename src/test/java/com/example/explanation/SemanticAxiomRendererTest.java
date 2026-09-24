package com.example.explanation;

import org.junit.jupiter.api.Test;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLDataFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SemanticAxiomRendererTest {
    private final OWLDataFactory dataFactory = OWLManager.getOWLDataFactory();

    @Test
    void deterministicManchesterUsesFullIris() {
        var axiom = dataFactory.getOWLSubClassOfAxiom(
                dataFactory.getOWLClass(IRI.create("https://one.example/Thing")),
                dataFactory.getOWLClass(IRI.create("https://two.example/Thing")));
        String first = SemanticAxiomRenderer.render(axiom);
        String second = SemanticAxiomRenderer.render(axiom);
        assertEquals(first, second);
        assertTrue(first.contains("<https://one.example/Thing>"));
        assertTrue(first.contains("<https://two.example/Thing>"));
    }

    @Test
    void functionalIdentityRoundTripsToManchester() throws Exception {
        String identity = "SubObjectPropertyOf(ObjectPropertyChain("
                + "<https://example.invalid/p> <https://example.invalid/q>) "
                + "<https://example.invalid/r>)";
        String rendering = SemanticAxiomRenderer.render(
                SemanticAxiomRenderer.parseFunctionalIdentity(identity));
        assertTrue(rendering.contains("o"));
        assertTrue(rendering.contains("SubPropertyOf"));
        assertNotEquals(identity, rendering);
    }
}
