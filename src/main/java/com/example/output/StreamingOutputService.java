// com/example/output/StreamingOutputService.java
package com.example.output;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicLong;
import java.util.List;


/**
 * Enhanced streaming output service with comprehensive CSV format and exact JSON structure
 */
public class StreamingOutputService implements OutputService {

    private static final Logger LOGGER = LoggerFactory.getLogger(StreamingOutputService.class);

    private final String outputDirectory;
    private final AtomicLong queryCounter = new AtomicLong(0);
    private final AtomicLong explanationCounter = new AtomicLong(0);
    private final ObjectMapper objectMapper = new ObjectMapper();  // ADD THIS FIELD

    private FileWriter queryWriter;
    private FileWriter explanationWriter;
    private long totalQueries = 0;
    private boolean isFirstExplanation = true;

    public StreamingOutputService(String outputDirectory) {
        this.outputDirectory = outputDirectory;
    }

    @Override
    public void initialize() throws IOException {
        LOGGER.info("Initializing StreamingOutputService with output directory: {}", outputDirectory);

        File outputDir = new File(outputDirectory);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        // Start each benchmark generation run from clean core outputs. Resuming by
        // appending is unsafe because an interrupted JSON file may be incomplete.
        File queryFile = new File(outputDir, "SPARQL_questions.csv");
        queryWriter = new FileWriter(queryFile, StandardCharsets.UTF_8, false);
        queryWriter.write("\"Task ID\",\"Root Entity\",\"Size of ontology TBox\",\"Size of ontology ABox\"," +
                "\"Task Type\",\"Answer Type\",\"SPARQL Query\",\"Predicate\",\"Answer\"," +
                "\"Min Tag Length\",\"Max Tag Length\"\n");
        queryWriter.flush();

        File explanationFile = new File(outputDir, "Explanations.json");
        explanationWriter = new FileWriter(explanationFile, StandardCharsets.UTF_8, false);
        explanationWriter.write("{\n");
        explanationWriter.flush();
        isFirstExplanation = true;

        LOGGER.info("Output files initialized: queries={}, explanations={}",
                queryFile.getPath(), explanationFile.getPath());
    }

    @Override
    public void writeQueryWithTags(String taskId, String query, String taskType, String answer, String explanation, String tags) {
        // Legacy method - redirect to comprehensive format with default tag lengths
        try {
            writeComprehensiveQuery(taskId, "Thing", 100, 50, taskType, "BIN", query, "predicate", answer, null, 1, 1);
        } catch (Exception e) {
            LOGGER.error("Error in legacy writeQueryWithTags: {}", e.getMessage());
        }
    }

    @Override
    public void writeExplanationWithTags(String key, String explanation, String tags) {
        try {
            long currentCount = explanationCounter.incrementAndGet();

            ObjectNode jsonNode = objectMapper.createObjectNode();
            jsonNode.put("explanation", explanation);
            jsonNode.put("tags", tags);
            jsonNode.put("timestamp", System.currentTimeMillis());

            String jsonString = objectMapper.writeValueAsString(jsonNode);

            synchronized (explanationWriter) {
                if (!isFirstExplanation) {
                    explanationWriter.write(",\n");
                }
                explanationWriter.write("\"" + escapeJSON(key) + "\": " + jsonString);
                explanationWriter.flush();
                isFirstExplanation = false;
            }

            if (currentCount % 1000 == 0) {
                logProgress("explanations", currentCount, 0);
            }

        } catch (IOException e) {
            LOGGER.error("Error writing explanation with tags: {}", key, e);
        }
    }

    @Override
    public void writeComprehensiveQuery(String taskId, String rootEntity, int tboxSize, int aboxSize,
                                        String taskType, String answerType, String sparqlQuery,
                                        String predicate, String answer, List<String> allAnswers,
                                        int minTagLength, int maxTagLength) {  // Updated parameters
        try {
            long currentCount = queryCounter.incrementAndGet();

            // For multi-choice queries, include all possible answers in the Answer column
            String finalAnswer = answer;
            if ("MC".equals(answerType) && allAnswers != null && !allAnswers.isEmpty()) {
                finalAnswer = String.join("; ", allAnswers);
            }

            // Updated format string - removed avg explanation count, changed to integers for tag lengths
            String csvLine = String.format("\"%s\",\"%s\",%d,%d,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%d,%d\n",
                    escapeCSV(taskId),
                    escapeCSV(rootEntity),
                    tboxSize,
                    aboxSize,
                    escapeCSV(taskType),
                    escapeCSV(answerType),
                    escapeCSV(sparqlQuery),
                    escapeCSV(predicate),
                    escapeCSV(finalAnswer),
                    minTagLength,
                    maxTagLength);

            synchronized (queryWriter) {
                queryWriter.write(csvLine);
                queryWriter.flush();
            }

            if (currentCount % 1000 == 0) {
                logProgress("queries", currentCount, totalQueries);
            }

        } catch (IOException e) {
            LOGGER.error("Error writing comprehensive query: {}", taskId, e);
        }
    }

    @Override
    public void writeExplanationWithComprehensiveFormat(String key, String comprehensiveExplanation) {
        try {
            long currentCount = explanationCounter.incrementAndGet();

            synchronized (explanationWriter) {
                if (!isFirstExplanation) {
                    explanationWriter.write(",\n");
                }
                explanationWriter.write(comprehensiveExplanation);
                explanationWriter.flush();
                isFirstExplanation = false;
            }

            if (currentCount % 1000 == 0) {
                logProgress("explanations", currentCount, 0);
            }

        } catch (IOException e) {
            LOGGER.error("Error writing comprehensive explanation: {}", key, e);
        }
    }

    private String escapeJSON(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private String escapeCSV(String value) {
        if (value == null) return "";
        return value.replace("\"", "\"\"").replace("\n", " ").replace("\r", " ");
    }

    @Override
    public void setTotalQueries(long total) {
        this.totalQueries = total;
        LOGGER.info("Expected total queries: {}", total);
    }

    @Override
    public void logProgress(String operation, long completed, long total) {
        if (total > 0) {
            double percentage = (completed * 100.0) / total;
            LOGGER.info("Progress {}: {}/{} ({:.1f}%)", operation, completed, total, percentage);
        } else {
            LOGGER.info("Progress {}: {} completed", operation, completed);
        }
    }

    @Override
    public void flush() {
        try {
            if (queryWriter != null) {
                queryWriter.flush();
            }
            if (explanationWriter != null) {
                explanationWriter.flush();
            }
        } catch (IOException e) {
            LOGGER.error("Error flushing output", e);
        }
    }

    @Override
    public void close() throws IOException {
        LOGGER.info("Closing StreamingOutputService. Final counts: queries={}, explanations={}",
                queryCounter.get(), explanationCounter.get());

        if (queryWriter != null) {
            try {
                queryWriter.close();
            } catch (IOException e) {
                LOGGER.error("Error closing query writer", e);
            } finally {
                queryWriter = null;
            }
        }

        if (explanationWriter != null) {
            try {
                explanationWriter.write("\n}");
                explanationWriter.close();
            } catch (IOException e) {
                LOGGER.error("Error closing explanation writer", e);
            } finally {
                explanationWriter = null;
            }
        }
    }
}
