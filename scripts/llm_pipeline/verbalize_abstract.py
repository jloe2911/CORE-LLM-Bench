# Import necessary libraries
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL
from pathlib import Path
import os
import re
import json
import gc
import psutil
import time

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
os.chdir(project_root)


def clean_entity_name(name):
    """Clean entity names to be more human-readable for any domain"""
    if not name:
        return ""

    # In abstracted ontologies, we often don't want to clean the names,
    # but we will keep basic underscore/hyphen replacement and spacing.
    name = name.replace("_", " ").replace("-", " ")

    # Convert camelCase to Title Case, which also helps with names like "Class5"
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

    # Capitalize properly and clean up spaces
    words = [word.capitalize() for word in name.split() if word]
    return " ".join(words)


def get_nice_label(g, entity):
    """Get a nice label for an entity from any domain"""
    if isinstance(entity, URIRef):
        uri_str = str(entity)
        if "#" in uri_str:
            name = uri_str.split("#")[-1]
        elif "/" in uri_str:
            name = uri_str.split("/")[-1]
        else:
            name = uri_str
        # For abstracted ontologies, the URI fragment is the name.
        return clean_entity_name(name)
    return None


class AbstractedOntologyVerbalizer:
    """
    Verbalizer for abstracted ontologies. It does not rely on linguistic cues
    in property names, instead using a consistent structural pattern.
    """

    def verbalize_relationship(self, g, subject_name, prop_uri, object_names):
        """
        Generates a clear, consistent sentence for any relationship.
        Example: "Subject has a 'Property' relationship with Object."
        """
        # Get the property's label using the graph
        clean_prop = get_nice_label(g, prop_uri)

        # Handle single or multiple objects
        if isinstance(object_names, list):
            if len(object_names) == 1:
                objects_text = object_names[0]
            elif len(object_names) == 2:
                objects_text = f"{object_names[0]} and {object_names[1]}"
            else:
                objects_text = f"{', '.join(object_names[:-1])}, and {object_names[-1]}"
        else:
            objects_text = object_names

        return f"{subject_name} has a '{clean_prop}' relationship with {objects_text}"


def create_simple_sentence(subject, verb, object_val):
    """Basic sentence creation without complex libraries."""
    return f"{subject} {verb} {object_val}."


def create_list_sentence(subject, verb, object_list):
    """Creates a sentence with a list of objects."""
    if not object_list:
        return ""
    if len(object_list) == 1:
        return f"{subject} {verb} {object_list[0]}."
    elif len(object_list) == 2:
        return f"{subject} {verb} {object_list[0]} and {object_list[1]}."
    else:
        return f"{subject} {verb} {', '.join(object_list[:-1])}, and {object_list[-1]}."


def _parse_restriction(g, restriction_node):
    """Helper function to parse an owl:Restriction and return a human-readable string."""
    on_prop_uri = g.value(restriction_node, OWL.onProperty)
    if not on_prop_uri:
        return None

    prop_name = get_nice_label(g, on_prop_uri)
    if not prop_name:
        return None

    # someValuesFrom
    some_values_from = g.value(restriction_node, OWL.someValuesFrom)
    if some_values_from:
        class_name = get_nice_label(g, some_values_from)
        if class_name:
            return (
                f"has some '{prop_name}' relationship with an instance of {class_name}"
            )

    # allValuesFrom
    all_values_from = g.value(restriction_node, OWL.allValuesFrom)
    if all_values_from:
        class_name = get_nice_label(g, all_values_from)
        if class_name:
            return (
                f"only has '{prop_name}' relationships with instances of {class_name}"
            )

    # hasSelf restriction
    has_self = g.value(restriction_node, OWL.hasSelf)
    if has_self and str(has_self).lower() == "true":
        return f"has a '{prop_name}' relationship with itself"

    # Qualified cardinality
    qual_cardinality = g.value(restriction_node, OWL.qualifiedCardinality)
    if qual_cardinality:
        on_class = g.value(restriction_node, OWL.onClass)
        class_name = get_nice_label(g, on_class) if on_class else "any class"
        return f"has exactly {qual_cardinality} '{prop_name}' relationship(s) with instances of {class_name}"

    # Maximum qualified cardinality
    max_qual_card = g.value(restriction_node, OWL.maxQualifiedCardinality)
    if max_qual_card:
        on_class = g.value(restriction_node, OWL.onClass)
        class_name = get_nice_label(g, on_class) if on_class else "any class"
        return f"has at most {max_qual_card} '{prop_name}' relationship(s) with instances of {class_name}"

    # Minimum qualified cardinality
    min_qual_card = g.value(restriction_node, OWL.minQualifiedCardinality)
    if min_qual_card:
        on_class = g.value(restriction_node, OWL.onClass)
        class_name = get_nice_label(g, on_class) if on_class else "any class"
        return f"has at least {min_qual_card} '{prop_name}' relationship(s) with instances of {class_name}"

    # Simple cardinality (without class qualification)
    cardinality = g.value(restriction_node, OWL.cardinality)
    if cardinality:
        return f"has exactly {cardinality} '{prop_name}' relationship(s)"

    max_cardinality = g.value(restriction_node, OWL.maxCardinality)
    if max_cardinality:
        return f"has at most {max_cardinality} '{prop_name}' relationship(s)"

    min_cardinality = g.value(restriction_node, OWL.minCardinality)
    if min_cardinality:
        return f"has at least {min_cardinality} '{prop_name}' relationship(s)"

    return None


def _parse_class_expression(g, bnode):
    """
    Recursively parses a complex class expression (BNode) and returns a human-readable string.
    Handles intersections, unions, and simple restrictions.
    """
    # Case 1: Intersection (e.g., a Person AND has a certain property)
    intersection_list_node = g.value(bnode, OWL.intersectionOf)
    if intersection_list_node:
        parts = parse_rdf_list(g, intersection_list_node)
        named_classes = []
        restrictions = []
        for part in parts:
            if isinstance(part, URIRef):
                named_classes.append(get_nice_label(g, part))
            elif isinstance(part, BNode):
                # Recursively parse nested expressions/restrictions
                restriction_desc = _parse_class_expression(g, part)
                if restriction_desc:
                    restrictions.append(restriction_desc)

        # Combine the parts into a coherent sentence
        description = ""
        if named_classes:
            description += " ".join(named_classes)
        if restrictions:
            # Phrase it as "a [Class] that also [restriction]"
            if named_classes:
                description += " that also "
            else:
                # If no named class, start with the restriction
                description += "a class that "
            description += " and ".join(restrictions)
        return description

    # Case 2: Union (e.g., a Person is a Man OR a Woman)
    union_list_node = g.value(bnode, OWL.unionOf)
    if union_list_node:
        parts = [
            get_nice_label(g, p)
            for p in parse_rdf_list(g, union_list_node)
            if get_nice_label(g, p)
        ]
        if len(parts) > 1:
            return f"either a {' or a '.join(parts)}"

    # Case 3: Complement (e.g., NOT a Person)
    complement_class = g.value(bnode, OWL.complementOf)
    if complement_class:
        complement_name = get_nice_label(g, complement_class)
        if complement_name:
            return f"the complement of {complement_name}"

    # Case 4: Simple Restriction (fallback)
    on_prop_uri = g.value(bnode, OWL.onProperty)
    if on_prop_uri:
        prop_name = get_nice_label(g, on_prop_uri)
        if not prop_name:
            return None

        # Has value restriction
        has_value = g.value(bnode, OWL.hasValue)
        if has_value:
            value_name = (
                get_nice_label(g, has_value)
                if isinstance(has_value, URIRef)
                else str(has_value)
            )
            return f"has '{prop_name}' value {value_name}"

        if val := g.value(bnode, OWL.someValuesFrom):
            if cls_name := get_nice_label(g, val):
                return f"has some '{prop_name}' relationship with an instance of {cls_name}"
        if val := g.value(bnode, OWL.allValuesFrom):
            if cls_name := get_nice_label(g, val):
                return (
                    f"only has '{prop_name}' relationships with instances of {cls_name}"
                )
        if card := g.value(bnode, OWL.qualifiedCardinality):
            if cls := g.value(bnode, OWL.onClass):
                if cls_name := get_nice_label(g, cls):
                    return f"has exactly {card} '{prop_name}' relationship(s) with an instance of {cls_name}"

    return None  # Fallback for unhandled expressions


def describe_class_with_domain_independence(
    g,
    cls,
    classes,
    subclass_relations,
    equivalent_class_relations,
    obj_properties,
    property_domains,
    property_ranges,
):
    """Generate natural language description of a class without domain assumptions"""
    descriptions = []
    cls_name = get_nice_label(g, cls)
    if not cls_name:
        return ""

    # Subclass relationships
    parents = [o for s, o in subclass_relations if s == cls]
    if parents:
        parent_names = [get_nice_label(g, p) for p in parents if get_nice_label(g, p)]
        if parent_names:
            descriptions.append(
                create_list_sentence(cls_name, "is a type of", parent_names)
            )

    # Equivalent and Complex Class Axioms
    axioms = equivalent_class_relations + [
        (s, o)
        for s, p, o in g.triples((cls, RDFS.subClassOf, None))
        if isinstance(o, BNode)
    ]

    for s, o in axioms:
        if s != cls:
            continue

        # Handle simple equivalent classes
        if isinstance(o, URIRef):
            if eq_name := get_nice_label(g, o):
                descriptions.append(
                    create_simple_sentence(cls_name, "is equivalent to", eq_name)
                )

        # Handle complex definitions (intersections, unions, restrictions)
        elif isinstance(o, BNode):
            expression_desc = _parse_class_expression(g, o)
            if expression_desc:
                # Determine the correct phrasing based on the axiom type
                axiom_type = (
                    "is defined as"
                    if (s, OWL.equivalentClass, o) in g
                    else "is a subclass of"
                )
                descriptions.append(f"{cls_name} {axiom_type} {expression_desc}.")

    # Disjoint Classes
    disjoint_classes = [
        get_nice_label(g, o)
        for s, p, o in g.triples((cls, OWL.disjointWith, None))
        if get_nice_label(g, o)
    ]
    if disjoint_classes:
        descriptions.append(
            f"{cls_name} is disjoint with {' and '.join(disjoint_classes)}."
        )

    # Disjoint Union handling (NEW)
    disjoint_union_node = g.value(cls, OWL.disjointUnionOf)
    if disjoint_union_node:
        disjoint_parts = [
            get_nice_label(g, p)
            for p in parse_rdf_list(g, disjoint_union_node)
            if get_nice_label(g, p)
        ]
        if disjoint_parts:
            descriptions.append(
                f"{cls_name} is the disjoint union of {', '.join(disjoint_parts)}."
            )

    # Key axioms (NEW)
    for s, p, o in g.triples((cls, OWL.hasKey, None)):
        key_properties = [
            get_nice_label(g, prop)
            for prop in parse_rdf_list(g, o)
            if get_nice_label(g, prop)
        ]
        if key_properties:
            descriptions.append(
                f"{cls_name} has key properties: {', '.join(key_properties)}."
            )

    # Fallback for simple classes with no other axioms
    if not descriptions:
        descriptions.append(
            create_simple_sentence(cls_name, "is", "a class in this ontology")
        )

    return " ".join(descriptions)


def parse_rdf_list(g, node, visited=None):
    """Recursively parses an RDF list and returns a Python list of its items."""
    if visited is None:
        visited = set()
    if node in visited:
        return []
    visited.add(node)

    items = []
    current = node
    while current and current != RDF.nil:
        first = g.value(current, RDF.first)
        if first:
            items.append(first)
        current = g.value(current, RDF.rest)
    return items


def describe_property_with_domain_independence(
    g, prop, obj_properties, property_domains, property_ranges
):
    """Generate natural language description of a property without domain assumptions"""
    descriptions = []
    prop_name = get_nice_label(g, prop)
    if not prop_name:
        return ""

    descriptions.append(f"{prop_name} is a relationship property.")

    # SubPropertyOf relationships - IMPROVED for consistency
    super_properties = []
    for s, p, o in g.triples((prop, RDFS.subPropertyOf, None)):
        super_prop_name = get_nice_label(g, o)
        if super_prop_name:
            super_properties.append(super_prop_name)

    if super_properties:
        if len(super_properties) == 1:
            descriptions.append(
                f"This property is a subproperty of {super_properties[0]}."
            )
        else:
            super_list = " and ".join(super_properties)
            descriptions.append(f"This property is a subproperty of {super_list}.")

    # Sub-properties (properties that are subproperties of this one) - IMPROVED
    sub_properties = []
    for s, p, o in g.triples((None, RDFS.subPropertyOf, prop)):
        sub_prop_name = get_nice_label(g, s)
        if sub_prop_name:
            sub_properties.append(sub_prop_name)

    if sub_properties:
        if len(sub_properties) == 1:
            descriptions.append(
                f"{sub_properties[0]} is a subproperty of this property."
            )
        else:
            sub_list = ", ".join(sub_properties[:-1]) + f" and {sub_properties[-1]}"
            descriptions.append(f"{sub_list} are subproperties of this property.")

    # Equivalent properties - MOVED to better position
    equiv_props = [
        get_nice_label(g, o)
        for s, p, o in g.triples((prop, OWL.equivalentProperty, None))
        if get_nice_label(g, o)
    ]
    if equiv_props:
        if len(equiv_props) == 1:
            descriptions.append(f"This property is equivalent to {equiv_props[0]}.")
        else:
            equiv_list = " and ".join(equiv_props)
            descriptions.append(f"This property is equivalent to {equiv_list}.")

    # Property characteristics - ENHANCED to catch all types
    characteristics = {o for s, p, o in g.triples((prop, RDF.type, None))}

    # All property characteristics including the missing ones - IMPROVED ORDER
    if OWL.FunctionalProperty in characteristics:
        descriptions.append(
            "This property is functional (each entity can have at most one value)."
        )
    if OWL.InverseFunctionalProperty in characteristics:
        descriptions.append(
            "This property is inverse functional (each value can be related to at most one entity)."
        )
    if OWL.SymmetricProperty in characteristics:
        descriptions.append(
            "This property is symmetric (if A relates to B, then B relates to A)."
        )
    if OWL.AsymmetricProperty in characteristics:
        descriptions.append(
            "This property is asymmetric (if A relates to B, then B cannot relate to A)."
        )
    if OWL.TransitiveProperty in characteristics:
        descriptions.append(
            "This property is transitive (it can form chains of relationships)."
        )
    if OWL.ReflexiveProperty in characteristics:
        descriptions.append(
            "This property is reflexive (every entity has this relationship with itself)."
        )
    if OWL.IrreflexiveProperty in characteristics:
        descriptions.append(
            "This property is irreflexive (an entity cannot have this relationship with itself)."
        )

    # Property chain axioms
    for s, p, o in g.triples((prop, OWL.propertyChainAxiom, None)):
        chain_names = [
            get_nice_label(g, item)
            for item in parse_rdf_list(g, o)
            if get_nice_label(g, item)
        ]
        if len(chain_names) > 1:
            chain_text = " followed by ".join(chain_names)
            descriptions.append(
                f"This property can be inferred from the chain of relationships: {chain_text}."
            )

    # Inverse properties
    for s, p, o in g.triples((prop, OWL.inverseOf, None)):
        if inv_name := get_nice_label(g, o):
            descriptions.append(f"It is the inverse of the {inv_name} property.")

    # Disjoint properties
    disjoint_props = [
        get_nice_label(g, o)
        for s, p, o in g.triples((prop, OWL.propertyDisjointWith, None))
        if get_nice_label(g, o)
    ]
    if disjoint_props:
        if len(disjoint_props) == 1:
            descriptions.append(f"This property is disjoint with {disjoint_props[0]}.")
        else:
            disjoint_list = " and ".join(disjoint_props)
            descriptions.append(f"This property is disjoint with {disjoint_list}.")

    # Domain and range information - IMPROVED to handle complex expressions
    domains = []
    for d in property_domains.get(prop, set()):
        if isinstance(d, URIRef):
            domain_name = get_nice_label(g, d)
            if domain_name:
                domains.append(domain_name)
        elif isinstance(d, BNode):
            # Handle complex domain expressions
            domain_desc = _parse_class_expression(g, d)
            if domain_desc:
                domains.append(f"({domain_desc})")

    ranges = []
    for r in property_ranges.get(prop, set()):
        if isinstance(r, URIRef):
            range_name = get_nice_label(g, r)
            if range_name:
                ranges.append(range_name)
        elif isinstance(r, BNode):
            # Handle complex range expressions
            range_desc = _parse_class_expression(g, r)
            if range_desc:
                ranges.append(f"({range_desc})")

    if domains and ranges:
        if len(domains) == 1 and len(ranges) == 1:
            descriptions.append(f"This property connects {domains[0]} to {ranges[0]}.")
        else:
            domain_text = " or ".join(domains)
            range_text = " or ".join(ranges)
            descriptions.append(
                f"This property connects {domain_text} to {range_text}."
            )
    elif domains:
        descriptions.append(f"This property has domain {' or '.join(domains)}.")
    elif ranges:
        descriptions.append(f"This property has range {' or '.join(ranges)}.")

    return " ".join(descriptions)


def describe_data_property_with_domain_independence(
    g, prop, property_domains, property_ranges
):
    """Generate natural language description of a data property"""
    descriptions = []
    prop_name = get_nice_label(g, prop)
    if not prop_name:
        return ""

    descriptions.append(f"{prop_name} is a data property.")

    # SubPropertyOf relationships - NEW/IMPROVED for data properties
    super_properties = []
    for s, p, o in g.triples((prop, RDFS.subPropertyOf, None)):
        super_prop_name = get_nice_label(g, o)
        if super_prop_name:
            super_properties.append(super_prop_name)

    if super_properties:
        if len(super_properties) == 1:
            descriptions.append(
                f"This property is a subproperty of {super_properties[0]}."
            )
        else:
            super_list = " and ".join(super_properties)
            descriptions.append(f"This property is a subproperty of {super_list}.")

    # Sub-properties - NEW for data properties
    sub_properties = []
    for s, p, o in g.triples((None, RDFS.subPropertyOf, prop)):
        sub_prop_name = get_nice_label(g, s)
        if sub_prop_name:
            sub_properties.append(sub_prop_name)

    if sub_properties:
        if len(sub_properties) == 1:
            descriptions.append(
                f"{sub_properties[0]} is a subproperty of this property."
            )
        else:
            sub_list = ", ".join(sub_properties[:-1]) + f" and {sub_properties[-1]}"
            descriptions.append(f"{sub_list} are subproperties of this property.")

    # Equivalent properties
    equiv_props = [
        get_nice_label(g, o)
        for s, p, o in g.triples((prop, OWL.equivalentProperty, None))
        if get_nice_label(g, o)
    ]
    if equiv_props:
        if len(equiv_props) == 1:
            descriptions.append(f"This property is equivalent to {equiv_props[0]}.")
        else:
            equiv_list = " and ".join(equiv_props)
            descriptions.append(f"This property is equivalent to {equiv_list}.")

    # Check for functional characteristic
    characteristics = {o for s, p, o in g.triples((prop, RDF.type, None))}
    if OWL.FunctionalProperty in characteristics:
        descriptions.append(
            "This property is functional (each entity can have at most one value)."
        )

    # Disjoint properties
    disjoint_props = [
        get_nice_label(g, o)
        for s, p, o in g.triples((prop, OWL.propertyDisjointWith, None))
        if get_nice_label(g, o)
    ]
    if disjoint_props:
        if len(disjoint_props) == 1:
            descriptions.append(f"This property is disjoint with {disjoint_props[0]}.")
        else:
            disjoint_list = " and ".join(disjoint_props)
            descriptions.append(f"This property is disjoint with {disjoint_list}.")

    # Domain and range - IMPROVED
    domains = []
    for d in property_domains.get(prop, set()):
        if isinstance(d, URIRef):
            domain_name = get_nice_label(g, d)
            if domain_name:
                domains.append(domain_name)

    ranges = []
    for r in property_ranges.get(prop, set()):
        if isinstance(r, URIRef):
            # Clean up datatype URIs for readability
            range_str = str(r)
            if "#" in range_str:
                clean_range = range_str.split("#")[-1]
            elif "/" in range_str:
                clean_range = range_str.split("/")[-1]
            else:
                clean_range = range_str
            ranges.append(clean_range)
        else:
            ranges.append(str(r))

    if domains and ranges:
        if len(domains) == 1 and len(ranges) == 1:
            descriptions.append(f"This property connects {domains[0]} to {ranges[0]}.")
        else:
            domain_text = " or ".join(domains)
            range_text = " or ".join(ranges)
            descriptions.append(
                f"This property connects {domain_text} to {range_text}."
            )
    elif domains:
        descriptions.append(
            f"This property applies to instances of {' or '.join(domains)}."
        )
    elif ranges:
        descriptions.append(f"Values are of type {' or '.join(ranges)}.")

    return " ".join(descriptions)


def get_all_individuals(g, classes, obj_properties):
    """Get all individuals from an ontology."""
    individuals = set()

    # Explicitly declared individuals
    for s, p, o in g.triples((None, RDF.type, OWL.NamedIndividual)):
        if isinstance(s, URIRef):
            individuals.add(s)

    # Individuals that are instances of classes
    for s, p, o in g.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and o in classes:
            individuals.add(s)

    # Individuals appearing as subjects/objects in object property statements
    for prop in obj_properties:
        for s, p, o in g.triples((None, prop, None)):
            if isinstance(s, URIRef) and s not in classes and s not in obj_properties:
                individuals.add(s)
            if isinstance(o, URIRef) and o not in classes and o not in obj_properties:
                individuals.add(o)

    # Check for individuals with data properties (including annotation properties)
    data_properties = {
        s
        for s, p, o in g.triples((None, RDF.type, OWL.DatatypeProperty))
        if isinstance(s, URIRef)
    }
    annotation_properties = {
        s
        for s, p, o in g.triples((None, RDF.type, OWL.AnnotationProperty))
        if isinstance(s, URIRef)
    }
    all_data_props = data_properties.union(annotation_properties)

    for prop in all_data_props:
        for s, p, o in g.triples((None, prop, None)):
            if isinstance(s, URIRef) and s not in classes and s not in obj_properties:
                individuals.add(s)

    # Look for individuals by URI pattern (for abstracted ontologies)
    for s, p, o in g:
        if isinstance(s, URIRef):
            uri_str = str(s)
            # Look for Individual pattern in abstracted ontologies
            if (
                "Individual" in uri_str
                and s not in classes
                and s not in obj_properties
                and s not in all_data_props
            ):
                individuals.add(s)

    return individuals


def describe_individual_with_domain_independence(
    g, ind, classes, obj_properties, data_properties, all_individuals=None
):
    """Generate natural language description of an individual for abstracted ontologies."""
    descriptions = []
    ind_name = get_nice_label(g, ind)
    if not ind_name:
        return ""

    verbalizer = AbstractedOntologyVerbalizer()

    # Get types/classes
    types = [
        get_nice_label(g, o)
        for s, p, o in g.triples((ind, RDF.type, None))
        if o != OWL.NamedIndividual and o in classes and get_nice_label(g, o)
    ]
    if types:
        descriptions.append(create_list_sentence(ind_name, "is an instance of", types))

    # Collect relationships
    outgoing_rels = {}
    for s, p, o in g.triples((ind, None, None)):
        if p in obj_properties and isinstance(o, URIRef):
            if p not in outgoing_rels:
                outgoing_rels[p] = []
            outgoing_rels[p].append(o)

    incoming_rels = {}
    for s, p, o in g.triples((None, None, ind)):
        if p in obj_properties and isinstance(s, URIRef) and s != ind:
            if p not in incoming_rels:
                incoming_rels[p] = []
            incoming_rels[p].append(s)

    # Data property relationships
    data_props = {}
    for s, p, o in g.triples((ind, None, None)):
        if p in data_properties:
            if p not in data_props:
                data_props[p] = []
            data_props[p].append(str(o))

    for prop_uri, values in data_props.items():
        prop_name = get_nice_label(g, prop_uri)
        if prop_name and values:
            if len(values) == 1:
                descriptions.append(f"{ind_name} has {prop_name} value {values[0]}.")
            else:
                descriptions.append(
                    f"{ind_name} has {prop_name} values: {', '.join(values)}."
                )

    # Verbalize outgoing relationships
    for prop_uri, obj_uris in outgoing_rels.items():
        obj_names = [get_nice_label(g, o) for o in obj_uris if get_nice_label(g, o)]
        if obj_names:
            desc = verbalizer.verbalize_relationship(g, ind_name, prop_uri, obj_names)
            descriptions.append(desc + ".")

    # Verbalize incoming relationships
    for prop_uri, subj_uris in incoming_rels.items():
        subj_names = [get_nice_label(g, s) for s in subj_uris if get_nice_label(g, s)]
        if subj_names:
            # We state the relationship from the subject's perspective
            for subj_name in subj_names:
                desc = verbalizer.verbalize_relationship(
                    g, subj_name, prop_uri, ind_name
                )
                descriptions.append(desc + ".")

    if not descriptions:
        descriptions.append(
            create_simple_sentence(ind_name, "is", "an individual in this ontology")
        )

    return " ".join(descriptions)


def collect_all_disjoint_unions(g, classes):
    """Collect all disjoint union relationships in the ontology"""
    disjoint_unions = []

    # Look for AllDisjointClasses axioms
    for s, p, o in g.triples((None, RDF.type, OWL.AllDisjointClasses)):
        members_node = g.value(s, OWL.members)
        if members_node:
            members = parse_rdf_list(g, members_node)
            member_names = [
                get_nice_label(g, m)
                for m in members
                if m in classes and get_nice_label(g, m)
            ]
            if len(member_names) > 1:
                disjoint_unions.append(
                    f"Classes {', '.join(member_names)} are all disjoint from each other"
                )

    return disjoint_unions


def verbalize_ontology(ontology_file_path, output_dir):
    """Verbalize a single ontology file and save as JSON - for abstracted ontologies"""
    print(f"Processing: {ontology_file_path}")
    g = Graph()
    try:
        g.parse(ontology_file_path)
    except Exception as e:
        print(f"Error loading ontology: {e}")
        return None

    try:
        # Extract structure
        classes = {
            s
            for s, p, o in g.triples((None, RDF.type, OWL.Class))
            if isinstance(s, URIRef)
        }
        obj_properties = {
            s
            for s, p, o in g.triples((None, RDF.type, OWL.ObjectProperty))
            if isinstance(s, URIRef)
        }
        data_properties = {
            s
            for s, p, o in g.triples((None, RDF.type, OWL.DatatypeProperty))
            if isinstance(s, URIRef)
        }
        annotation_properties = {
            s
            for s, p, o in g.triples((None, RDF.type, OWL.AnnotationProperty))
            if isinstance(s, URIRef)
        }
        data_properties = data_properties.union(annotation_properties)
        individuals = get_all_individuals(g, classes, obj_properties)

        domain_type = "relational" if len(obj_properties) > 10 else "general"

        subclass_relations = [
            (s, o)
            for s, p, o in g.triples((None, RDFS.subClassOf, None))
            if s in classes and o in classes
        ]
        equivalent_class_relations = [
            (s, o)
            for s, p, o in g.triples((None, OWL.equivalentClass, None))
            if s in classes
        ]

        # IMPROVED: Collect both simple and complex domain/range information
        property_domains = {}
        property_ranges = {}

        for p in obj_properties:
            domains = set()
            ranges = set()
            for s, pred, o in g.triples((p, RDFS.domain, None)):
                domains.add(o)
            for s, pred, o in g.triples((p, RDFS.range, None)):
                ranges.add(o)
                property_domains[p] = domains
                property_ranges[p] = ranges

        data_property_domains = {}
        data_property_ranges = {}

        for p in data_properties:
            domains = set()
            ranges = set()
            for s, pred, o in g.triples((p, RDFS.domain, None)):
                domains.add(o)
            for s, pred, o in g.triples((p, RDFS.range, None)):
                ranges.add(o)
            data_property_domains[p] = domains
            data_property_ranges[p] = ranges

        # NEW: Collect global axioms and disjoint unions
        disjoint_unions = collect_all_disjoint_unions(g, classes)

        # Create verbalization data
        verbalization_data = {
            "ontology": {
                "name": Path(ontology_file_path).stem.replace("_abs", ""),
                "structuralType": domain_type,
                "triples": len(g),
                "description": f"This {domain_type} ontology contains information about {len(individuals)} individuals, {len(classes)} classes, {len(obj_properties)} object properties, and {len(data_properties)} data properties.",
            },
            "classes": [],
            "objectProperties": [],
            "dataProperties": [],
            "individuals": [],
        }

        # NEW: Add global axioms if present
        if disjoint_unions:
            verbalization_data["ontology"]["globalAxioms"] = disjoint_unions

        # Generate descriptions
        for cls in sorted(classes, key=lambda x: get_nice_label(g, x) or ""):
            class_desc = describe_class_with_domain_independence(
                g,
                cls,
                classes,
                subclass_relations,
                equivalent_class_relations,
                obj_properties,
                property_domains,
                property_ranges,
            )
            verbalization_data["classes"].append(
                {"classLabel": get_nice_label(g, cls), "description": class_desc}
            )

        for prop in sorted(obj_properties, key=lambda x: get_nice_label(g, x) or ""):
            prop_desc = describe_property_with_domain_independence(
                g, prop, obj_properties, property_domains, property_ranges
            )
            verbalization_data["objectProperties"].append(
                {"propertyLabel": get_nice_label(g, prop), "description": prop_desc}
            )

        for prop in sorted(data_properties, key=lambda x: get_nice_label(g, x) or ""):
            prop_desc = describe_data_property_with_domain_independence(
                g, prop, data_property_domains, data_property_ranges
            )
            verbalization_data["dataProperties"].append(
                {"propertyLabel": get_nice_label(g, prop), "description": prop_desc}
            )

        for ind in sorted(individuals, key=lambda x: get_nice_label(g, x) or ""):
            ind_desc = describe_individual_with_domain_independence(
                g, ind, classes, obj_properties, data_properties, individuals
            )
            verbalization_data["individuals"].append(
                {"individualLabel": get_nice_label(g, ind), "description": ind_desc}
            )

        # Save JSON
        output_file = output_dir / f"{Path(ontology_file_path).stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(verbalization_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved verbalization to: {output_file}")
        return output_file

    except Exception as e:
        import traceback

        print(f"❌ Error during verbalization: {e}\n{traceback.format_exc()}")
        return None
    finally:
        g.close()
        del g
        gc.collect()


def main():
    """Main function to process abstracted ontology files"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verbalize ontology files to natural language JSON (domain-independent)"
    )
    parser.add_argument(
        "--input-dir",
        default="output/OWL2Bench/2hop/abstracted/abstracted_ontologies",
        help="Input directory containing ontology files",
    )
    parser.add_argument(
        "--output-dir",
        default="output/verbalized_ontologies/OWL2Bench_2hop/abstracted",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.ttl",
        help="File pattern to match (e.g., *.ttl, *.owl, *.rdf)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of files to process in each batch",
    )
    parser.add_argument(
        "--memory-threshold",
        type=int,
        default=80,
        help="Memory usage threshold (%) to trigger cleanup pause",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    ontology_files = list(input_dir.glob(args.file_pattern))
    if not ontology_files:
        print(f"No files matching '{args.file_pattern}' found in {input_dir}")
        return

    print(f"Found {len(ontology_files)} abstracted ontology files to process.")

    successful, failed = 0, 0
    for i, ontology_file in enumerate(ontology_files):
        # Memory management
        if i > 0 and i % args.batch_size == 0:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > args.memory_threshold:
                print(f"Memory usage at {memory_percent}%, pausing for cleanup...")
                gc.collect()
                time.sleep(2)

        if verbalize_ontology(ontology_file, output_dir):
            successful += 1
        else:
            failed += 1

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(ontology_files)} files processed")

    print(f"\n{'=' * 60}\nProcessing completed!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")


if __name__ == "__main__":
    main()
