package com.example.processing;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;

class SmallOntologiesProcessorTest {

    @Test
    void pizzaMembershipChoosesNonDomainConceptRepresentative() {
        Optional<String> representative = SmallOntologiesProcessor.selectRepresentativePositive(
                "rdf:type", List.of("DomainConcept", "Pizza"), true);

        assertEquals(Optional.of("Pizza"), representative);
    }

    @Test
    void pizzaMembershipOmitsDomainConceptOnlyGroup() {
        Optional<String> representative = SmallOntologiesProcessor.selectRepresentativePositive(
                "rdf:type", List.of("DomainConcept"), true);

        assertTrue(representative.isEmpty());
    }

    @Test
    void nonPizzaAndNonMembershipSelectionRemainUnchanged() {
        assertEquals(
                Optional.of("DomainConcept"),
                SmallOntologiesProcessor.selectRepresentativePositive(
                        "rdf:type", List.of("DomainConcept"), false));
        assertEquals(
                Optional.of("DomainConcept"),
                SmallOntologiesProcessor.selectRepresentativePositive(
                        "hasCategory", List.of("DomainConcept"), true));
    }

    @Test
    void pizzaMembershipNeverUsesDomainConceptAsBinaryTarget() {
        assertFalse(SmallOntologiesProcessor.isEligibleBinaryTarget(
                "rdf:type", "DomainConcept", true));
        assertTrue(SmallOntologiesProcessor.isEligibleBinaryTarget(
                "rdf:type", "Pizza", true));
        assertTrue(SmallOntologiesProcessor.isEligibleBinaryTarget(
                "hasCategory", "DomainConcept", true));
    }

    @Test
    void answerProjectionPreservesDistinctIriIdentity() {
        assertEquals(
                List.of("Alpha", "Beta"),
                SmallOntologiesProcessor.identityPreservingLocalNames(List.of(
                        "https://example.org/a#Beta",
                        "https://example.org/a#Alpha")));

        assertThrows(
                IllegalStateException.class,
                () -> SmallOntologiesProcessor.identityPreservingLocalNames(List.of(
                        "https://example.org/a#Same",
                        "https://example.org/b#Same")));
    }
}
