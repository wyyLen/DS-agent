# DSAgent Core

**Framework-Agnostic Data Science Agent Mechanisms**

DSAgent Core is a standalone library that provides reusable components for building intelligent data science agents. It was extracted from the DSAgent project to enable integration with any agent framework (MetaGPT, LangChain, AutoGen, etc.).

## Features

### 🔍 Experience Retrieval

#### Text Experience Retrieval
- **BM25-based matching** for finding relevant textual experiences
- Configurable similarity thresholds
- Support for metadata filtering
- Fast and lightweight implementation

#### Workflow Experience Retrieval
- **Graph-based matching** for workflow/plan similarity
- Considers both task types and DAG structure
- Customizable weights for different similarity aspects
- Perfect for finding similar solution patterns

### 🌳 Tree Search

#### Language Agent Tree Search (LATS)
- **Monte Carlo Tree Search** variant for autonomous exploration
- UCB-based node selection for exploration-exploitation balance
- Customizable action generation, evaluation, and termination
- Framework-agnostic state representation

## Installation

```bash
# Install from source
cd dsagent-main
pip install -e dsagent_core

# Or install specific components
pip install -e dsagent_core[retrieval]  # Only retrieval
pip install -e dsagent_core[search]     # Only tree search
pip install -e dsagent_core[all]        # Everything
```

## Quick Start

### Text Experience Retrieval

```python
from pathlib import Path
from dsagent_core.retrieval import TextExperienceRetriever

# Initialize retriever
retriever = TextExperienceRetriever(
    experience_path=Path("experiences.json"),
    top_k=5,
    min_score_threshold=10.0
)

# Retrieve relevant experiences
result = retriever.retrieve(
    query="How to handle missing values in pandas?",
    top_k=3
)

# Use the results
for exp in result.experiences:
    print(f"Score: {exp.score:.2f}")
    print(f"Content: {exp.content}")
```

### Workflow Experience Retrieval

```python
from dsagent_core.retrieval import WorkflowExperienceRetriever

# Initialize retriever
retriever = WorkflowExperienceRetriever(
    experience_path=Path("workflow_experiences.json"),
    top_k=5
)

# Define your workflow
workflow = [
    {
        "task_id": "1",
        "instruction": "Load and explore dataset",
        "task_type": "pda",
        "dependent_task_ids": []
    },
    {
        "task_id": "2",
        "instruction": "Calculate correlations",
        "task_type": "statistical analysis",
        "dependent_task_ids": ["1"]
    }
]

# Find similar workflows
result = retriever.retrieve(query=workflow, top_k=3)

for exp in result.experiences:
    print(f"Similarity: {exp.score:.2f}")
    print(f"Tasks: {exp.metadata['num_tasks']}")
```

### Tree Search

```python
from dsagent_core.search import (
    TreeSearchEngine,
    ActionGenerator,
    StateEvaluator,
    TerminationChecker
)

# Implement your custom components
class MyActionGenerator(ActionGenerator):
    def generate_actions(self, state, context):
        # Return list of possible next states
        return [...]

class MyEvaluator(StateEvaluator):
    def evaluate(self, state, context):
        # Return score for state
        return 0.8

class MyTerminationChecker(TerminationChecker):
    def is_terminal(self, state, context):
        # Check if goal reached
        return False

# Create search engine
engine = TreeSearchEngine(
    action_generator=MyActionGenerator(),
    state_evaluator=MyEvaluator(),
    termination_checker=MyTerminationChecker(),
    max_depth=10,
    max_iterations=100
)

# Perform search
root = engine.search(initial_state="start")
best_path = engine.get_best_path(root)
```

## Integration with Frameworks

### MetaGPT

```python
from dsagent_core.adapters import MetaGPTAdapter
from pathlib import Path

# Create adapter
adapter = MetaGPTAdapter(
    text_exp_path=Path("exp_bank/plan_exp.json"),
    workflow_exp_path=Path("exp_bank/workflow_exp2_clean.json")
)

# Use in MetaGPT agent
query = "Analyze correlation in dataset"
result = adapter.retrieve_text_experiences(query, top_k=3)

# Format for LLM prompt
formatted = adapter.format_experiences_for_prompt(result)
# ... use in your MetaGPT action ...
```

### LangChain (Coming Soon)

```python
from dsagent_core.adapters import LangChainAdapter

adapter = LangChainAdapter(...)
# ... integration code ...
```

### AutoGen (Coming Soon)

```python
from dsagent_core.adapters import AutoGenAdapter

adapter = AutoGenAdapter(...)
# ... integration code ...
```

## Experience File Format

### Text Experiences (`plan_exp.json`)

```json
[
    {
        "task": "Task description or question",
        "solution": "Detailed solution or approach",
        "metadata": {
            "task_type": "data_analysis",
            "domain": "pandas"
        }
    }
]
```

### Workflow Experiences (`workflow_exp2_clean.json`)

```json
[
    {
        "workflow": [
            {
                "task_id": "1",
                "instruction": "Task description",
                "task_type": "pda",
                "dependent_task_ids": []
            }
        ],
        "exp": "Summary of this workflow pattern",
        "task": "Original problem description"
    }
]
```

## Architecture

```
dsagent_core/
├── retrieval/
│   ├── base.py              # Abstract interfaces
│   ├── text_retriever.py    # BM25-based retrieval
│   └── workflow_retriever.py # Graph-based retrieval
├── search/
│   └── tree_search.py       # LATS implementation
└── adapters/
    ├── metagpt_adapter.py   # MetaGPT integration
    ├── langchain_adapter.py # (Coming soon)
    └── autogen_adapter.py   # (Coming soon)
```

## Configuration Options

### TextExperienceRetriever

```python
retriever = TextExperienceRetriever(
    experience_path=Path("..."),
    top_k=5,                    # Number of results
    bm25_k1=1.5,               # BM25 parameter
    bm25_b=0.75,               # BM25 parameter
    min_score_threshold=0.0    # Minimum score filter
)
```

### WorkflowExperienceRetriever

```python
retriever = WorkflowExperienceRetriever(
    experience_path=Path("..."),
    top_k=5,
    type_weight=0.7,           # Weight for task type similarity
    structure_weight=0.3,      # Weight for graph structure
    min_similarity_threshold=0.3
)
```

### TreeSearchEngine

```python
engine = TreeSearchEngine(
    action_generator=...,
    state_evaluator=...,
    termination_checker=...,
    max_depth=10,              # Maximum tree depth
    max_iterations=100,        # Maximum search iterations
    exploration_weight=1.414   # UCB exploration constant
)
```

## Advanced Usage

### Custom Experience Formats

```python
from dsagent_core.retrieval.base import ExperienceEntry

# Create custom experience
exp = ExperienceEntry(
    content="My custom experience",
    metadata={
        "custom_field": "value",
        "tags": ["tag1", "tag2"]
    }
)

# Add to retriever
retriever.add_experience(exp)
```

### Filtering Results

```python
# Filter by metadata
result = retriever.retrieve(
    query="some query",
    filters={
        "task_type": "machine_learning",
        "domain": "sklearn"
    }
)
```

### Statistics and Monitoring

```python
# Get retriever statistics
stats = retriever.get_statistics()
print(f"Total experiences: {stats['total_experiences']}")
print(f"Average content length: {stats['avg_content_length']}")

# Get search statistics
stats = engine.get_statistics(root_node)
print(f"Nodes explored: {stats['total_nodes']}")
print(f"Max depth: {stats['max_depth_reached']}")
```

## Performance

- **Text Retrieval**: ~10-50ms for 1000 experiences
- **Workflow Retrieval**: ~100-500ms for 500 workflows  
- **Tree Search**: Depends on action generation cost

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Citation

If you use DSAgent Core in your research, please cite:

```bibtex
@software{dsagent_core,
  title={DSAgent Core: Framework-Agnostic Data Science Agent Mechanisms},
  author={DSAgent Team},
  year={2026},
  url={https://github.com/yourusername/dsagent}
}
```

## Support

- **Documentation**: [Full docs](https://dsagent-core.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/dsagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/dsagent/discussions)

## Roadmap

- [x] Text experience retrieval
- [x] Workflow experience retrieval
- [x] Tree search mechanism
- [x] MetaGPT adapter
- [ ] LangChain adapter
- [ ] AutoGen adapter
- [ ] Semantic embedding support
- [ ] Distributed search
- [ ] Web UI for experience management
