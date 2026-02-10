| Compared Method   | Metric            |   Normality p-value | Normal Distribution   | Test Used            |   Test p-value | Significance    |   Mean Δ (Proposed - Compared) |
|:------------------|:------------------|--------------------:|:----------------------|:---------------------|---------------:|:----------------|-------------------------------:|
| Naive_LLM_qwen    | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         4.984  |
| Naive_LLM_qwen    | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         5.2946 |
| Naive_LLM_qwen    | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         4.8437 |
| Naive_LLM_qwen    | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         5.1743 |
| Naive_LLM_qwen    | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         4.2285 |
| Naive_LLM_gpt     | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         3.316  |
| Naive_LLM_gpt     | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         3.43   |
| Naive_LLM_gpt     | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.93   |
| Naive_LLM_gpt     | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         3.288  |
| Naive_LLM_gpt     | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    1e-10       | Significant     |                         1.83   |
| Zeroshot          | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    1.83618e-05 | Significant     |                         1.04   |
| Zeroshot          | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    4.5266e-06  | Significant     |                         1.108  |
| Zeroshot          | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    1.67984e-05 | Significant     |                         0.99   |
| Zeroshot          | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    1.25783e-05 | Significant     |                         1.042  |
| Zeroshot          | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    7.7462e-06  | Significant     |                         1.068  |
| vectorrag         | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.56   |
| vectorrag         | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.42   |
| vectorrag         | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         1.904  |
| vectorrag         | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.262  |
| vectorrag         | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    0.000926683 | Significant     |                         0.944  |
| Openie            | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.04   |
| Openie            | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.124  |
| Openie            | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         1.776  |
| Openie            | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.156  |
| Openie            | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    0.000371524 | Significant     |                         0.974  |
| kggen             | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    0.174044    | Not Significant |                         0.35   |
| kggen             | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    0.130545    | Not Significant |                         0.352  |
| kggen             | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    0.159397    | Not Significant |                         0.326  |
| kggen             | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    0.158185    | Not Significant |                         0.348  |
| kggen             | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    0.183679    | Not Significant |                         0.332  |
| graphrag_global   | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.702  |
| graphrag_global   | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.58   |
| graphrag_global   | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.694  |
| graphrag_global   | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.558  |
| graphrag_global   | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.562  |
| graphrag_local    | Completeness      |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.782  |
| graphrag_local    | Accuracy          |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.782  |
| graphrag_local    | Knowledgeability  |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.722  |
| graphrag_local    | Relevance         |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.436  |
| graphrag_local    | Logical_coherence |                   0 | No                    | Wilcoxon signed-rank |    0           | Significant     |                         2.43   |