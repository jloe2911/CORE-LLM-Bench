// com/example/processing/SmallOntologiesProcessor.java
package com.example.processing;

import com.example.config.ProcessingConfiguration;
import com.example.ontology.OntologyService;
import com.example.reasoning.ReasoningService;
import com.example.explanation.ComprehensiveExplanationService;
import com.example.explanation.ExplanationPath;
import com.example.explanation.ExplanationFormatter;
import com.example.explanation.EnhancedExplanationTagger;
import com.example.query.QueryGenerationService;
import com.example.output.OutputService;
import com.example.util.OntologyUtils;
import com.example.util.URIUtils;

import org.semanticweb.owlapi.model.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * IMPROVED: Sequential processor for handling multiple small ontologies
 * Processes one ontology at a time to prevent memory issues
 */
public class SmallOntologiesProcessor implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(SmallOntologiesProcessor.class);
    private static final int MAX_ROOT_CONTEXTS_PER_BASE_QUERY = 3;

    // Services
    private final OntologyService ontologyService;
    private final ReasoningService reasoningService;
    private final QueryGenerationService queryService;
    private final OutputService outputService;
    private final ProcessingConfiguration config;
    private final EnhancedExplanationTagger tagger;
    private final PerformanceTracker performanceTracker;

    // Counters for tracking across all ontologies
    private final AtomicLong totalOntologiesProcessed = new AtomicLong(0);
    private final AtomicLong totalInferencesProcessed = new AtomicLong(0);
    private final AtomicLong totalQueriesGenerated = new AtomicLong(0);
    private final AtomicLong totalBinaryQueries = new AtomicLong(0);
    private final AtomicLong totalMultiChoiceQueries = new AtomicLong(0);

    // For tracking MC queries across current ontology only
    private Set<String> currentOntologyMCQueries;
    private String taskIdHopPrefix = "1hop";

    public SmallOntologiesProcessor(OntologyService ontologyService,
                                    ReasoningService reasoningService,
                                    QueryGenerationService queryService,
                                    OutputService outputService,
                                    ProcessingConfiguration config) {
        this.ontologyService = ontologyService;
        this.reasoningService = reasoningService;
        this.queryService = queryService;
        this.outputService = outputService;
        this.config = config;
        this.tagger = new EnhancedExplanationTagger();
        this.performanceTracker = new PerformanceTracker();

        LOGGER.info("SmallOntologiesProcessor initialized for SEQUENTIAL processing");
    }

    public ProcessingResult processSmallOntologies(String ontologiesDirectory) {
        ProcessingResult result = new ProcessingResult();
        performanceTracker.start("total_processing");

        try {
            LOGGER.info("Starting SEQUENTIAL processing of small ontologies from: {}", ontologiesDirectory);
            taskIdHopPrefix = inferHopPrefix(ontologiesDirectory);
            LOGGER.info("Using '{}' task ID prefix", taskIdHopPrefix);

            // Step 1: Initialize output service
            outputService.initialize();

            // Step 2: Get list of ontology files (don't load them all at once)
            performanceTracker.start("file_discovery");
            List<File> ontologyFiles = discoverOntologyFiles(ontologiesDirectory);
            performanceTracker.end("file_discovery");

            LOGGER.info("Discovered {} ontology files", ontologyFiles.size());
            if (ontologyFiles.isEmpty()) {
                LOGGER.warn("No ontology files found in directory: {}", ontologiesDirectory);
                result.setError("No ontology files found in directory");
                return result;
            }

            // Step 3: Process each ontology file sequentially
            performanceTracker.start("sequential_processing");
            processOntologyFilesSequentially(ontologyFiles, result);
            performanceTracker.end("sequential_processing");

            // Step 4: Finalize results
            finalizeResults(result);

        } catch (Exception e) {
            LOGGER.error("Error during small ontologies processing", e);
            result.setError("Processing failed: " + e.getMessage());
        } finally {
            performanceTracker.end("total_processing");
            performanceTracker.logSummary();
        }

        return result;
    }

    /**
     * NEW: Discover ontology files without loading them
     */
    private List<File> discoverOntologyFiles(String directoryPath) {
        File directory = new File(directoryPath);
        if (!directory.exists() || !directory.isDirectory()) {
            throw new RuntimeException("Directory does not exist or is not a directory: " + directoryPath);
        }

        File[] files = directory.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".owl") ||
                        name.toLowerCase().endsWith(".ttl") ||
                        name.toLowerCase().endsWith(".rdf") ||
                        name.toLowerCase().endsWith(".n3"));

        return files != null ? Arrays.asList(files) : new ArrayList<>();
    }

    /**
     * REPLACED: Process ontology files one by one to avoid memory issues
     */
    private void processOntologyFilesSequentially(List<File> ontologyFiles, ProcessingResult result) {
        for (int i = 0; i < ontologyFiles.size(); i++) {
            File ontologyFile = ontologyFiles.get(i);

            try {
                LOGGER.info("Processing file {}/{}: {}",
                        i + 1, ontologyFiles.size(), ontologyFile.getName());

                // Process single ontology file and immediately write outputs
                boolean success = processSingleOntologyFile(ontologyFile, result);

                if (success) {
                    totalOntologiesProcessed.incrementAndGet();
                }

                // Force garbage collection after each ontology
                System.gc();

                // Log progress every 10 files
                if ((i + 1) % 10 == 0) {
                    long processed = totalOntologiesProcessed.get();
                    LOGGER.info("Progress: {}/{} files processed ({}%)",
                            processed, ontologyFiles.size(), formatDouble((processed * 100.0) / ontologyFiles.size(), 1));
                    logMemoryUsage();
                }

            } catch (Exception e) {
                LOGGER.warn("Error processing ontology file {}: {}", ontologyFile.getName(), e.getMessage());
                result.addError("Failed to process file " + ontologyFile.getName() + ": " + e.getMessage());
            }
        }
    }

    /**
     * NEW: Process a single ontology file completely and write outputs immediately
     */
    private boolean processSingleOntologyFile(File ontologyFile, ProcessingResult result) {
        OWLOntology ontology = null;
        ComprehensiveExplanationService explanationService = null;

        try {
            // Reset MC query tracking for this ontology
            currentOntologyMCQueries = new HashSet<>();

            // Load single ontology
            ontology = ontologyService.loadOntology(ontologyFile);
            LOGGER.debug("Loaded ontology: {} with {} axioms",
                    ontologyFile.getName(), ontology.getAxiomCount());

            // UPDATED: Extract root entity from TTL filename (not from ontology IRI)
            String rootEntity = extractRootEntityFromFilename(ontologyFile);
            LOGGER.debug("Extracted root entity from filename: {}", rootEntity);

            // Calculate and set TBox and ABox sizes for this ontology
            int calculatedTBoxSize = OntologyUtils.calculateTBoxSize(ontology);
            int calculatedABoxSize = OntologyUtils.calculateABoxSize(ontology);
            calculateOntologyStats(calculatedTBoxSize, calculatedABoxSize);

            // Initialize reasoner for this ontology
            reasoningService.initializeReasoner(ontology);

            if (!reasoningService.isConsistent()) {
                LOGGER.warn("Inconsistent ontology detected: {}", ontologyFile.getName());
                result.addWarning("Inconsistent ontology: " + ontologyFile.getName());
                return false;
            }

            Map<String, Set<ExplanationPath>> ontologyInferences;
            if (config.isGenerateExplanations()) {
                explanationService = new ComprehensiveExplanationService(
                        reasoningService.getReasoner(), ontology,
                        config.getMaxExplanationsPerInference());
                ontologyInferences = extractInferencesWithExplanations(ontology, explanationService, rootEntity);
            } else {
                ontologyInferences = extractInferencesWithoutExplanations(ontology, rootEntity);
            }

            // Process and write inferences using the instance fields
            processAndWriteInferences(ontologyInferences, ontology, tboxSize, aboxSize, rootEntity, result);

            totalInferencesProcessed.addAndGet(ontologyInferences.size());
            return true;

        } catch (Exception e) {
            LOGGER.error("Error processing ontology file: {}", ontologyFile.getName(), e);
            return false;
        } finally {
            // CRITICAL: Clean up resources immediately
            cleanupResources(explanationService);
        }
    }

    private int tboxSize = 0;
    private int aboxSize = 0;

    /**
     * Set the TBox and ABox sizes for the current ontology
     */
    public synchronized void calculateOntologyStats(int tboxSize, int aboxSize) {
        this.tboxSize = tboxSize;
        this.aboxSize = aboxSize;
        LOGGER.info("Set ontology sizes - TBox: {}, ABox: {}", tboxSize, aboxSize);
    }

    /**
     * UPDATED: Extract inferences - get INFERRED triples for queries, but explain ASSERTED triples
     */
    private Map<String, Set<ExplanationPath>> extractInferencesWithExplanations(
            OWLOntology ontology, ComprehensiveExplanationService explanationService, String rootEntity) {

        Map<String, Set<ExplanationPath>> inferences = new HashMap<>();
        Set<OWLNamedIndividual> individuals = selectIndividualsForExplanation(ontology, rootEntity);
        int totalIndividuals = ontology.getIndividualsInSignature().size();

        LOGGER.info("Extracting explanations for {} of {} individuals", individuals.size(), totalIndividuals);

        int processedIndividuals = 0;
        for (OWLNamedIndividual individual : individuals) {
            try {
                // Extract class assertions - use INFERRED for queries, but explain how ASSERTED ones could be inferred
                extractClassAssertionInferences(individual, ontology, explanationService, inferences);

                // Extract property assertions - use INFERRED for queries, but explain how ASSERTED ones could be inferred
                extractPropertyAssertionInferences(individual, ontology, explanationService, inferences);

            } catch (Exception e) {
                LOGGER.debug("Error processing individual {}: {}", individual, e.getMessage());
            }

            processedIndividuals++;
            if (processedIndividuals == 1 || processedIndividuals % 25 == 0 || processedIndividuals == individuals.size()) {
                LOGGER.info(
                        "Explanation extraction progress: {}/{} individuals, {} explained inferences collected",
                        processedIndividuals, individuals.size(), inferences.size());
            }
        }

        LOGGER.info("Extracted {} explained inferences from ontology", inferences.size());
        return inferences;
    }

    private Map<String, Set<ExplanationPath>> extractInferencesWithoutExplanations(
            OWLOntology ontology, String rootEntity) {

        Map<String, Set<ExplanationPath>> inferences = new LinkedHashMap<>();
        Set<OWLNamedIndividual> individuals = selectIndividualsForExplanation(ontology, rootEntity);
        int totalIndividuals = ontology.getIndividualsInSignature().size();

        LOGGER.info("Extracting inferred triples without explanations for {} of {} individuals",
                individuals.size(), totalIndividuals);

        int processedIndividuals = 0;
        for (OWLNamedIndividual individual : individuals) {
            try {
                extractClassAssertionInferencesWithoutExplanations(individual, inferences);
                extractPropertyAssertionInferencesWithoutExplanations(individual, ontology, inferences);
            } catch (Exception e) {
                LOGGER.debug("Error processing individual {}: {}", individual, e.getMessage());
            }

            processedIndividuals++;
            if (processedIndividuals == 1 || processedIndividuals % 25 == 0 || processedIndividuals == individuals.size()) {
                LOGGER.info(
                        "Inference extraction progress: {}/{} individuals, {} inferred triples collected",
                        processedIndividuals, individuals.size(), inferences.size());
            }
        }

        LOGGER.info("Extracted {} inferred triples from ontology without explanations", inferences.size());
        return inferences;
    }

    private Set<OWLNamedIndividual> selectIndividualsForExplanation(OWLOntology ontology, String rootEntity) {
        List<OWLNamedIndividual> allIndividuals = ontology.getIndividualsInSignature().stream()
                .sorted(Comparator.comparing(individual -> OntologyUtils.getShortForm(individual)))
                .collect(Collectors.toList());

        if (config.isFocusRootIndividualOnly()) {
            Optional<OWLNamedIndividual> rootIndividual = findRootIndividual(allIndividuals, rootEntity);
            if (rootIndividual.isPresent()) {
                LOGGER.info("Focused explanation extraction on root individual: {}",
                        OntologyUtils.getShortForm(rootIndividual.get()));
                return new LinkedHashSet<>(Collections.singletonList(rootIndividual.get()));
            }

            LOGGER.warn("Could not identify root individual for '{}'; falling back to all individuals", rootEntity);
        }

        int maxIndividuals = config.getMaxIndividualsPerOntology();
        if (maxIndividuals > 0 && allIndividuals.size() > maxIndividuals) {
            LOGGER.info("Limiting explanation extraction to first {} individuals out of {}",
                    maxIndividuals, allIndividuals.size());
            return new LinkedHashSet<>(allIndividuals.subList(0, maxIndividuals));
        }

        return new LinkedHashSet<>(allIndividuals);
    }

    private Optional<OWLNamedIndividual> findRootIndividual(List<OWLNamedIndividual> individuals, String rootEntity) {
        if (rootEntity == null || rootEntity.isBlank()) {
            return Optional.empty();
        }

        String normalizedRoot = rootEntity.trim();

        Optional<OWLNamedIndividual> exactMatch = individuals.stream()
                .filter(individual -> OntologyUtils.getShortForm(individual).equals(normalizedRoot))
                .findFirst();
        if (exactMatch.isPresent()) {
            return exactMatch;
        }

        return individuals.stream()
                .filter(individual -> normalizedRoot.endsWith("_" + OntologyUtils.getShortForm(individual)))
                .max(Comparator.comparingInt(individual -> OntologyUtils.getShortForm(individual).length()));
    }

    private Set<ExplanationPath> limitExplanationPaths(Set<ExplanationPath> paths) {
        int maxExplanations = config.getMaxExplanationsPerInference();
        if (maxExplanations <= 0 || paths.size() <= maxExplanations) {
            return paths;
        }

        return paths.stream()
                .limit(maxExplanations)
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    /**
     * UPDATED: Use INFERRED types for query generation, but explain ASSERTED types
     */
    private void extractClassAssertionInferences(OWLNamedIndividual individual,
                                                 OWLOntology ontology,
                                                 ComprehensiveExplanationService explanationService,
                                                 Map<String, Set<ExplanationPath>> inferences) {
        try {
            // Get INFERRED types from reasoner (for query generation)
            Set<OWLClass> inferredTypes = reasoningService.getReasoner()
                    .getTypes(individual, false).getFlattened();

            // Get ASSERTED types from ontology (for explanations)
            Set<OWLClass> assertedTypes = ontology.getClassAssertionAxioms(individual).stream()
                    .map(ax -> ax.getClassExpression())
                    .filter(expr -> !expr.isAnonymous())
                    .map(expr -> expr.asOWLClass())
                    .collect(Collectors.toSet());

            // Process INFERRED types for queries, but generate explanations for ASSERTED types
            for (OWLClass inferredClass : inferredTypes) {
                if (OntologyUtils.isOwlThing(inferredClass)) continue;

                String tripleKey = OntologyUtils.createTripleKey(
                        OntologyUtils.getShortForm(individual),
                        "rdf:type",
                        OntologyUtils.getShortForm(inferredClass)
                );

                // Generate explanations: "How could this inferred class membership be derived?"
                Set<ExplanationPath> paths = explanationService.findExplanationPathsLikeProtege(individual, inferredClass);

                if (!paths.isEmpty()) {
                    inferences.put(tripleKey, limitExplanationPaths(paths));

                    LOGGER.debug("Found {} explanation paths for INFERRED type: {} rdf:type {}",
                            paths.size(), OntologyUtils.getShortForm(individual), OntologyUtils.getShortForm(inferredClass));
                }
            }

        } catch (Exception e) {
            LOGGER.debug("Error extracting class assertions for {}: {}", individual, e.getMessage());
        }
    }

    private void extractClassAssertionInferencesWithoutExplanations(
            OWLNamedIndividual individual,
            Map<String, Set<ExplanationPath>> inferences) {
        try {
            Set<OWLClass> inferredTypes = reasoningService.getReasoner()
                    .getTypes(individual, false).getFlattened();

            for (OWLClass inferredClass : inferredTypes) {
                if (OntologyUtils.isOwlThing(inferredClass)) continue;

                String tripleKey = OntologyUtils.createTripleKey(
                        OntologyUtils.getShortForm(individual),
                        "rdf:type",
                        OntologyUtils.getShortForm(inferredClass)
                );
                inferences.put(tripleKey, Collections.emptySet());
            }
        } catch (Exception e) {
            LOGGER.debug("Error extracting class assertions for {}: {}", individual, e.getMessage());
        }
    }

    /**
     * UPDATED: Use INFERRED properties for query generation, but explain ASSERTED properties
     */
    private void extractPropertyAssertionInferences(OWLNamedIndividual individual,
                                                    OWLOntology ontology,
                                                    ComprehensiveExplanationService explanationService,
                                                    Map<String, Set<ExplanationPath>> inferences) {
        try {
            // Get ALL object properties in the ontology
            Set<OWLObjectProperty> properties = ontology.getObjectPropertiesInSignature();

            for (OWLObjectProperty property : properties) {
                // Get INFERRED values from reasoner (for query generation)
                Set<OWLNamedIndividual> inferredValues = reasoningService.getReasoner()
                        .getObjectPropertyValues(individual, property).getFlattened();

                // Process INFERRED values for queries
                for (OWLNamedIndividual inferredValue : inferredValues) {
                    String tripleKey = OntologyUtils.createTripleKey(
                            OntologyUtils.getShortForm(individual),
                            OntologyUtils.getShortForm(property),
                            OntologyUtils.getShortForm(inferredValue)
                    );

                    // Generate explanations: "How could this inferred property assertion be derived?"
                    Set<ExplanationPath> paths = explanationService.findPropertyAssertionPaths(
                            individual, property, inferredValue);

                    if (!paths.isEmpty()) {
                        inferences.put(tripleKey, limitExplanationPaths(paths));

                        LOGGER.debug("Found {} explanation paths for property: {} {} {}",
                                paths.size(),
                                OntologyUtils.getShortForm(individual),
                                OntologyUtils.getShortForm(property),
                                OntologyUtils.getShortForm(inferredValue));
                    }
                }
            }
        } catch (Exception e) {
            LOGGER.debug("Error extracting property assertions for {}: {}", individual, e.getMessage());
        }
    }

    private void extractPropertyAssertionInferencesWithoutExplanations(
            OWLNamedIndividual individual,
            OWLOntology ontology,
            Map<String, Set<ExplanationPath>> inferences) {
        try {
            Set<OWLObjectProperty> properties = ontology.getObjectPropertiesInSignature();

            for (OWLObjectProperty property : properties) {
                Set<OWLNamedIndividual> inferredValues = reasoningService.getReasoner()
                        .getObjectPropertyValues(individual, property).getFlattened();

                for (OWLNamedIndividual inferredValue : inferredValues) {
                    String tripleKey = OntologyUtils.createTripleKey(
                            OntologyUtils.getShortForm(individual),
                            OntologyUtils.getShortForm(property),
                            OntologyUtils.getShortForm(inferredValue)
                    );
                    inferences.put(tripleKey, Collections.emptySet());
                }
            }
        } catch (Exception e) {
            LOGGER.debug("Error extracting property assertions for {}: {}", individual, e.getMessage());
        }
    }

    private void processAndWriteInferences(Map<String, Set<ExplanationPath>> inferences,
                                           OWLOntology ontology, int tboxSize, int aboxSize,
                                           String rootEntity, ProcessingResult result) {

        LOGGER.info("Writing {} inferences to CSV/JSON outputs", inferences.size());

        String ontologyName = extractOntologyName(ontology);

        // Group inferences by subject-predicate for MC queries
        Map<String, Map<String, Set<String>>> subjectPredicateObjects = groupInferencesForMCQueries(inferences);

        long binaryQueries = 0;
        long negativeBinaryQueries = 0;
        long multiChoiceQueries = 0;
        CandidateObjectPools candidateObjects = collectCandidateObjects(ontology, inferences);
        Map<String, Set<String>> usedNegativeObjects = new HashMap<>();

        int writtenInferences = 0;
        for (Map.Entry<String, Set<ExplanationPath>> entry : inferences.entrySet()) {
            String tripleKey = entry.getKey();
            Set<ExplanationPath> paths = entry.getValue();

            try {
                String[] parts = OntologyUtils.parseTripleKey(tripleKey);
                if (parts.length != 3) continue;

                String subject = parts[0];
                String predicate = parts[1];
                String object = parts[2];
                String inferenceKey = createRootScopedQueryKey(rootEntity, tripleKey);

                // Keep limited root-context diversity without allowing highly
                // repeated ontology patterns to dominate the generated dataset.
                if (!GlobalQueryTracker.markQueryProcessedWithContextLimit(
                        inferenceKey, tripleKey, rootEntity, ontologyName, MAX_ROOT_CONTEXTS_PER_BASE_QUERY)) {
                    LOGGER.debug("Skipping duplicate inference: {} (first seen in {})",
                            inferenceKey, GlobalQueryTracker.getFirstOntology(inferenceKey));
                    continue;
                }

                String taskType = "rdf:type".equals(predicate) ? "Membership" : "Property Assertion";

                // Calculate tag statistics instead of explanation statistics
                int[] tagStats = calculateTagStats(paths);

                if (shouldGenerateBinaryForGroup(subject, predicate, object, subjectPredicateObjects)) {
                    // 2. Write one representative positive binary query per subject-predicate group.
                    String binaryTaskId = URIUtils.generateBinaryTaskId(
                            taskIdHopPrefix, rootEntity, subject, predicate, object);
                    GlobalQueryTracker.addTaskId(inferenceKey, binaryTaskId);

                    String binaryQuery = String.format("ASK WHERE { <%s> <%s> <%s> }",
                            URIUtils.getFullURI(subject), URIUtils.getFullURI(predicate), URIUtils.getFullURI(object));

                    outputService.writeComprehensiveQuery(
                            binaryTaskId, rootEntity, tboxSize, aboxSize, taskType, "BIN",
                            binaryQuery, predicate,
                            "TRUE", null, tagStats[0], tagStats[1]
                    );
                    binaryQueries++;

                    // 2b. Write one paired negative binary query for the same subject-predicate group.
                    String negativeObject = chooseNegativeObject(
                            subject, predicate, object, subjectPredicateObjects, candidateObjects,
                            inferences.keySet(), usedNegativeObjects);

                    if (negativeObject != null) {
                        String negativeBaseQueryKey = "NEG|" + OntologyUtils.createTripleKey(subject, predicate, negativeObject);
                        String negativeQueryKey = createRootScopedQueryKey(rootEntity, negativeBaseQueryKey);

                        if (GlobalQueryTracker.markQueryProcessedWithContextLimit(
                                negativeQueryKey, negativeBaseQueryKey, rootEntity, ontologyName,
                                MAX_ROOT_CONTEXTS_PER_BASE_QUERY)) {
                            String negativeTaskId = generateNegativeTaskId(rootEntity, subject, predicate, negativeObject);

                            String negativeBinaryQuery = String.format("ASK WHERE { <%s> <%s> <%s> }",
                                    URIUtils.getFullURI(subject), URIUtils.getFullURI(predicate), URIUtils.getFullURI(negativeObject));

                            outputService.writeComprehensiveQuery(
                                    negativeTaskId, rootEntity, tboxSize, aboxSize, taskType, "BIN",
                                    negativeBinaryQuery, predicate,
                                    "FALSE", null, tagStats[0], tagStats[1]
                            );
                            GlobalQueryTracker.addTaskId(negativeQueryKey, negativeTaskId);
                            negativeBinaryQueries++;
                        }
                    }
                }

                // 3. Write one open-ended SELECT query per subject-predicate group.
                if (shouldGenerateMultiChoiceQuery(subject, predicate, subjectPredicateObjects)) {
                    String multiBaseQueryKey = "MC|" + subject + "|" + predicate;
                    String multiQueryKey = createRootScopedQueryKey(rootEntity, multiBaseQueryKey);
                    if (GlobalQueryTracker.markQueryProcessedWithContextLimit(
                            multiQueryKey, multiBaseQueryKey, rootEntity, ontologyName,
                            MAX_ROOT_CONTEXTS_PER_BASE_QUERY)) {
                        Set<String> allObjectsSet = subjectPredicateObjects.get(subject).get(predicate);
                        List<String> allAnswers = allObjectsSet.stream()
                                .sorted()
                                .collect(Collectors.toList());

                        String multiTaskId = URIUtils.generateTaskId(taskIdHopPrefix, rootEntity, subject, predicate, "MC");
                        GlobalQueryTracker.addTaskId(multiQueryKey, multiTaskId);
                        for (String answerObject : allAnswers) {
                            String answerTripleKey = OntologyUtils.createTripleKey(subject, predicate, answerObject);
                            String answerInferenceKey = createRootScopedQueryKey(rootEntity, answerTripleKey);
                            GlobalQueryTracker.addTaskId(answerInferenceKey, multiTaskId);
                        }

                        // MC query is SELECT - doesn't specify the object
                        String multiQuery = String.format("SELECT ?x WHERE { <%s> <%s> ?x }",
                                URIUtils.getFullURI(subject), URIUtils.getFullURI(predicate));

                        outputService.writeComprehensiveQuery(
                                multiTaskId, rootEntity, tboxSize, aboxSize, taskType, "MC",
                                multiQuery, predicate,
                                object, allAnswers,
                                tagStats[0], tagStats[1]  // Updated to use tag stats
                        );
                        multiChoiceQueries++;
                    }
                }

                if (config.isGenerateExplanations()) {
                    // Write comprehensive explanation to JSON after generating task IDs.
                    String comprehensiveExplanation = ExplanationFormatter.generateExactJSONFormat(
                            inferenceKey, tripleKey, paths, tagger);
                    outputService.writeExplanationWithComprehensiveFormat(inferenceKey, comprehensiveExplanation);
                }

            } catch (Exception e) {
                LOGGER.warn("Error processing inference {}: {}", tripleKey, e.getMessage());
                result.addWarning("Failed to process inference: " + tripleKey);
            }

            writtenInferences++;
            if (writtenInferences == 1 || writtenInferences % 100 == 0 || writtenInferences == inferences.size()) {
                LOGGER.info(
                        "Output progress: {}/{} inferences written, {} queries generated so far for ontology {}",
                        writtenInferences,
                        inferences.size(),
                        binaryQueries + negativeBinaryQueries + multiChoiceQueries,
                        ontologyName);
            }
        }

        // Update counters and flush
        totalBinaryQueries.addAndGet(binaryQueries + negativeBinaryQueries);
        totalMultiChoiceQueries.addAndGet(multiChoiceQueries);
        totalQueriesGenerated.addAndGet(binaryQueries + negativeBinaryQueries + multiChoiceQueries);

        LOGGER.debug("Wrote {} positive binary queries, {} negative binary queries, and {} MC queries for ontology {}",
                binaryQueries, negativeBinaryQueries, multiChoiceQueries, ontologyName);

        outputService.flush();
    }

    private String createRootScopedQueryKey(String rootEntity, String queryKey) {
        return rootEntity + "||" + queryKey;
    }

    // Helper method to extract ontology name
    private String extractOntologyName(OWLOntology ontology) {
        try {
            Optional<IRI> ontologyIRI = ontology.getOntologyID().getOntologyIRI();
            if (ontologyIRI.isPresent()) {
                return URIUtils.getLocalName(ontologyIRI.get().toString());
            }
        } catch (Exception e) {
            LOGGER.debug("Could not extract ontology name: {}", e.getMessage());
        }
        return "unknown";
    }

    private boolean shouldGenerateBinaryForGroup(String subject, String predicate, String object,
                                                 Map<String, Map<String, Set<String>>> subjectPredicateObjects) {
        Map<String, Set<String>> predicateObjects = subjectPredicateObjects.get(subject);
        if (predicateObjects == null) return false;

        Set<String> objects = predicateObjects.get(predicate);
        if (objects == null || objects.isEmpty()) return false;

        Comparator<String> representativeComparator = Comparator
                .comparingInt((String candidate) -> negativeCandidatePriority(predicate, candidate))
                .thenComparing(Comparator.naturalOrder());

        return objects.stream()
                .sorted(representativeComparator)
                .findFirst()
                .map(object::equals)
                .orElse(false);
    }

    // Updated to check against subject-predicate combinations
    private boolean shouldGenerateMultiChoiceQuery(String subject, String predicate,
                                                   Map<String, Map<String, Set<String>>> subjectPredicateObjects) {
        Map<String, Set<String>> predicateObjects = subjectPredicateObjects.get(subject);
        if (predicateObjects == null) return false;

        Set<String> objects = predicateObjects.get(predicate);
        return objects != null && !objects.isEmpty();
    }

    /**
     * Collect available object terms so negative binary questions can use
     * plausible same-kind alternatives instead of obvious class/individual
     * mismatches.
     */
    private CandidateObjectPools collectCandidateObjects(
            OWLOntology ontology,
            Map<String, Set<ExplanationPath>> inferences) {
        Set<String> inferredClassObjects = inferences.keySet().stream()
                .map(OntologyUtils::parseTripleKey)
                .filter(parts -> parts.length == 3)
                .filter(parts -> "rdf:type".equals(parts[1]))
                .map(parts -> parts[2])
                .distinct()
                .collect(Collectors.toCollection(TreeSet::new));

        Set<String> inferredIndividualObjects = inferences.keySet().stream()
                .map(OntologyUtils::parseTripleKey)
                .filter(parts -> parts.length == 3)
                .filter(parts -> !"rdf:type".equals(parts[1]))
                .map(parts -> parts[2])
                .distinct()
                .collect(Collectors.toCollection(TreeSet::new));

        Set<String> signatureClasses = ontology.getClassesInSignature().stream()
                .map(OntologyUtils::getShortForm)
                .collect(Collectors.toCollection(TreeSet::new));

        Set<String> signatureIndividuals = ontology.getIndividualsInSignature().stream()
                .map(OntologyUtils::getShortForm)
                .collect(Collectors.toCollection(TreeSet::new));

        inferredClassObjects.addAll(signatureClasses);
        inferredIndividualObjects.addAll(signatureIndividuals);

        return new CandidateObjectPools(
                new ArrayList<>(inferredClassObjects),
                new ArrayList<>(inferredIndividualObjects));
    }

    /**
     * Choose an object that makes the same subject-predicate ASK query false.
     * Preference is given to objects seen with the same predicate elsewhere, then
     * to any object in the current ontology's generated inferences.
     */
    private String chooseNegativeObject(String subject, String predicate, String object,
                                        Map<String, Map<String, Set<String>>> subjectPredicateObjects,
                                        CandidateObjectPools candidateObjects,
                                        Set<String> positiveTripleKeys,
                                        Map<String, Set<String>> usedNegativeObjects) {
        Set<String> trueObjectsForSubjectPredicate = subjectPredicateObjects
                .getOrDefault(subject, Collections.emptyMap())
                .getOrDefault(predicate, Collections.emptySet());
        String subjectPredicateKey = subject + "|" + predicate;
        Set<String> usedForSubjectPredicate = usedNegativeObjects
                .computeIfAbsent(subjectPredicateKey, key -> new HashSet<>());

        Comparator<String> negativeCandidateComparator = Comparator
                .comparingInt((String candidate) -> negativeCandidatePriority(predicate, candidate))
                .thenComparing(Comparator.naturalOrder());

        List<String> samePredicateCandidates = subjectPredicateObjects.values().stream()
                .map(predicateObjects -> predicateObjects.getOrDefault(predicate, Collections.emptySet()))
                .flatMap(Set::stream)
                .distinct()
                .sorted(negativeCandidateComparator)
                .collect(Collectors.toList());

        List<String> sameKindCandidates = "rdf:type".equals(predicate)
                ? candidateObjects.classObjects
                : candidateObjects.individualObjects;

        String candidate = findNegativeObjectCandidate(
                subject, predicate, trueObjectsForSubjectPredicate, samePredicateCandidates,
                positiveTripleKeys, usedForSubjectPredicate);
        if (candidate != null) {
            usedForSubjectPredicate.add(candidate);
            return candidate;
        }

        candidate = findNegativeObjectCandidate(
                subject, predicate, trueObjectsForSubjectPredicate, sameKindCandidates,
                positiveTripleKeys, usedForSubjectPredicate);
        if (candidate != null) {
            usedForSubjectPredicate.add(candidate);
            return candidate;
        }

        return null;
    }

    private String findNegativeObjectCandidate(String subject, String predicate,
                                               Set<String> trueObjectsForSubjectPredicate,
                                               List<String> candidates,
                                               Set<String> positiveTripleKeys,
                                               Set<String> usedForSubjectPredicate) {
        for (String candidate : candidates) {
            if (trueObjectsForSubjectPredicate.contains(candidate) ||
                    usedForSubjectPredicate.contains(candidate)) {
                continue;
            }

            String candidateTripleKey = OntologyUtils.createTripleKey(subject, predicate, candidate);
            if (!positiveTripleKeys.contains(candidateTripleKey)) {
                return candidate;
            }
        }

        return null;
    }

    private int negativeCandidatePriority(String predicate, String candidate) {
        if (!"rdf:type".equals(predicate)) {
            return 0;
        }

        String localName = URIUtils.getLocalName(candidate);
        if (Set.of("Thing", "NamedIndividual", "DomainEntity", "Person").contains(localName)) {
            return 1;
        }

        return 0;
    }

    private static class CandidateObjectPools {
        private final List<String> classObjects;
        private final List<String> individualObjects;

        private CandidateObjectPools(List<String> classObjects, List<String> individualObjects) {
            this.classObjects = classObjects;
            this.individualObjects = individualObjects;
        }
    }

    private String generateNegativeTaskId(String rootEntity, String subject, String predicate, String negativeObject) {
        return URIUtils.generateBinaryTaskId(taskIdHopPrefix, rootEntity, subject, predicate, negativeObject)
                .replace("-BIN", "-NEG-BIN");
    }

    private String inferHopPrefix(String ontologiesDirectory) {
        if (ontologiesDirectory == null) {
            return "1hop";
        }

        String normalized = ontologiesDirectory.toLowerCase(Locale.ROOT);
        if (normalized.contains("2hop")) {
            return "2hop";
        }
        if (normalized.contains("1hop")) {
            return "1hop";
        }

        return "1hop";
    }

    /**
     * UPDATED: Extract root entity directly from TTL filename
     */
    private String extractRootEntityFromFilename(File ontologyFile) {
        try {
            String fileName = ontologyFile.getName();

            // Remove file extension (.ttl, .owl, .rdf, .n3)
            String rootEntity = fileName.replaceAll("\\.(ttl|owl|rdf|n3)$", "");

            LOGGER.debug("Extracted root entity '{}' from filename '{}'", rootEntity, fileName);
            return rootEntity;

        } catch (Exception e) {
            LOGGER.warn("Could not extract root entity from filename {}: {}",
                    ontologyFile.getName(), e.getMessage());

            // Fallback: use filename without extension
            String fileName = ontologyFile.getName();
            int lastDot = fileName.lastIndexOf('.');
            return lastDot > 0 ? fileName.substring(0, lastDot) : fileName;
        }
    }

    // NEW METHOD - Classify task type based on predicate
    private String getTaskType(String predicate) {
        if ("rdf:type".equals(predicate)) {
            return "Membership";
        } else {
            return "Property Assertion";
        }
    }

    /**
     * NEW: Group inferences by subject-predicate to prepare for MC queries
     */
    private Map<String, Map<String, Set<String>>> groupInferencesForMCQueries(
            Map<String, Set<ExplanationPath>> inferences) {

        Map<String, Map<String, Set<String>>> grouped = new HashMap<>();

        for (String tripleKey : inferences.keySet()) {
            String[] parts = OntologyUtils.parseTripleKey(tripleKey);
            if (parts.length != 3) continue;

            String subject = parts[0];
            String predicate = parts[1];
            String object = parts[2];

            grouped.computeIfAbsent(subject, k -> new HashMap<>())
                    .computeIfAbsent(predicate, k -> new HashSet<>())
                    .add(object);
        }

        return grouped;
    }

    /**
     * Calculate tag length statistics from explanation paths
     */
    private int[] calculateTagStats(Set<ExplanationPath> paths) {
        if (paths.isEmpty()) {
            return new int[]{0, 0};
        }

        int minTagLength = Integer.MAX_VALUE;
        int maxTagLength = 0;

        for (ExplanationPath path : paths) {
            String tag = tagger.tagExplanation(path);
            int tagLength = tag != null ? tag.length() : 0;

            minTagLength = Math.min(minTagLength, tagLength);
            maxTagLength = Math.max(maxTagLength, tagLength);
        }

        // Handle case where all paths have empty tags
        if (minTagLength == Integer.MAX_VALUE) {
            minTagLength = 0;
        }

        return new int[]{minTagLength, maxTagLength};
    }

    /**
     * NEW: Clean up resources after processing each ontology
     */
    private void cleanupResources(ComprehensiveExplanationService explanationService) {
        try {
            if (reasoningService != null) {
                reasoningService.close(); // This disposes the reasoner
            }
        } catch (Exception e) {
            LOGGER.warn("Error closing reasoning service", e);
        }

        explanationService = null;
        System.gc(); // Force garbage collection
    }

    /**
     * NEW: Log current memory usage
     */
    private void logMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        long maxMemory = runtime.maxMemory();

        LOGGER.info("Memory Usage: {} MB used, {} MB free, {} MB total, {} MB max",
                formatDouble(usedMemory / (1024.0 * 1024.0), 2),
                formatDouble(freeMemory / (1024.0 * 1024.0), 2),
                formatDouble(totalMemory / (1024.0 * 1024.0), 2),
                formatDouble(maxMemory / (1024.0 * 1024.0), 2));
    }

    private String formatDouble(double value, int decimals) {
        return String.format(Locale.ROOT, "%." + decimals + "f", value);
    }

    /**
     * UPDATED: Finalize results with new counters
     */
    private void finalizeResults(ProcessingResult result) {
        result.setProcessingTimeMs(performanceTracker.getDuration("total_processing"));
        result.setTotalInferences(totalInferencesProcessed.get());
        result.addProcessedQueries(totalQueriesGenerated.get());
        result.setBinaryQueries(totalBinaryQueries.get());
        result.setMultiChoiceQueries(totalMultiChoiceQueries.get());

        // Calculate memory usage
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        result.setMemoryUsedMB(usedMemory / (1024.0 * 1024.0));

        result.setSuccess(true);

        LOGGER.info("Sequential processing completed successfully!");
        LOGGER.info("Final Statistics:");
        LOGGER.info("  Processed ontologies: {}", totalOntologiesProcessed.get());
        LOGGER.info("  Total inferences: {}", totalInferencesProcessed.get());
        LOGGER.info("  Total queries generated: {}", totalQueriesGenerated.get());
        LOGGER.info("  Binary queries: {}", totalBinaryQueries.get());
        LOGGER.info("  Multi-choice queries: {}", totalMultiChoiceQueries.get());
    }

    @Override
    public void close() throws Exception {
        LOGGER.info("Closing SmallOntologiesProcessor...");

        // Close services
        try {
            if (outputService != null) {
                outputService.close();
            }
        } catch (Exception e) {
            LOGGER.warn("Error closing output service", e);
        }

        try {
            if (reasoningService != null) {
                reasoningService.close();
            }
        } catch (Exception e) {
            LOGGER.warn("Error closing reasoning service", e);
        }

        try {
            if (ontologyService != null) {
                ontologyService.close();
            }
        } catch (Exception e) {
            LOGGER.warn("Error closing ontology service", e);
        }

        LOGGER.info("SmallOntologiesProcessor closed successfully");
    }
}
