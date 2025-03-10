
```markdown
| Model with Setup                              | TPQ-2019 | TPQ_2020 |
|-----------------------------------------------|----------|----------|
|ATLAS - 64 shot (Table 3 in current paper)     | 74.68    | 40.93    |
|TempRALM - ATLAS - 64 shot (Table 3 in paper)  | 75.78    | 68.44    |
|-----------------------------------------------|----------|----------|
|GPT - closed book (Table 4 in current paper)   | 68.75    | 65.62    |
|-----------------------------------------------|----------|----------|
|GPT - RAG - Embeddings-small                   | 34.375   | 12.5     |
|TEMPRALM-GPT-Embeddings-small                  | 59.375   | 43.75    |
|-----------------------------------------------|----------|----------|
|GPT - RAG - Embeddings-large                   | 43.8     | 43.75    |
|TEMPRALM - GPT-RAG -Embeddings-Large           | 75       | 56.25    |
|-----------------------------------------------|----------|----------|
|GPT - RAG - Embeddings-Large - Few Shot        | 90.625   | 50       |
|TEMPRALM - GPT-RAG -Embeddings-Large - Few Shot| 93.75    | 65.63    |
```


```
| Model with Setup                          | TPQ-2019 | TPQ_2020 |
|-------------------------------------------|----------|----------|
| ATLAS - 64 shot                           | 85.714   | 53.571   |
| TempRALM - ATLAS - 64 shot                | 92.857   | 85.714   |
|-------------------------------------------|----------|----------|
| GPT - closed book                         | 57.14    | 17.8     |
|-------------------------------------------|----------|----------|
| GPT - RAG - Embeddings-small              | TBD      | TBD      |
| TEMPRALM-GPT-Embeddings-small             | TBD      | TBD      |
|-------------------------------------------|----------|----------|
| GPT - RAG - Embeddings-large              | 65.38    | 57.69    |
| TEMPRALM - GPT-RAG -Embeddings-Large      | 73.07    | 76.92    |
|-------------------------------------------|----------|----------|
| GPT - RAG - Embeddings-Large - Few Shot   | 73.07    | 42.3     |
| TEMPRALM-GPT-RAG-Embeddings-Large-Few Shot| 84.61    | 76.92    |
```


```
| Model                                          | Correct Passage present in prompt to GPT | GPT answered correctly | EM     |
|------------------------------------------------|------------------------------------------|------------------------|--------|
| GPT - RAG                                      | 18/32                                    | 14/32                  | 43.75  |
| GPT - RAG with Few Shot                        | 18/32                                    | 16/32                  | 50     |
| GPT - TempRALM - large embeddings              | 28/32                                    | 18/32                  | 56.25  |
| GPT - TempRALM - large embeddings with Few shot| 28/32                                    | 21/32                  | 65.625 |
```