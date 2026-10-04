package com.example.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class URIUtilsTest {
    @Test
    void preservesAbsoluteIriAndExpandsStandardRdfPrefix() {
        assertEquals("https://example.org#Entity", URIUtils.getFullURI("https://example.org#Entity"));
        assertEquals(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                URIUtils.getFullURI("rdf:type"));
    }

    @Test
    void refusesToInventNamespaceForLocalName() {
        assertThrows(IllegalArgumentException.class, () -> URIUtils.getFullURI("Pizza"));
    }
}
