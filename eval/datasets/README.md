# Evaluation Datasets

This directory contains evaluation datasets for testing StoryLand AI quality across different scenarios.

## Dataset Files

### books_v1.json

**Version**: 1.0
**Created**: 2026-02-11
**Purpose**: Initial evaluation dataset with 10 diverse book queries

**Coverage**:
- 10 different books spanning multiple genres and time periods
- Variety of user preferences (budget, pace, accessibility, family travel)
- Geographical diversity (Europe, Americas, Middle East, Russia)
- Different trip types (luxury, budget, family, spiritual)

**Books Included**:
1. Pride and Prejudice - Jane Austen (Classic Romance)
2. 1984 - George Orwell (Dystopian Fiction)
3. The Great Gatsby - F. Scott Fitzgerald (American Classic)
4. Les Misérables - Victor Hugo (Historical Fiction)
5. War and Peace - Leo Tolstoy (Historical Fiction)
6. The Da Vinci Code - Dan Brown (Mystery Thriller)
7. Harry Potter - J.K. Rowling (Fantasy)
8. The Girl with the Dragon Tattoo - Stieg Larsson (Crime Thriller)
9. A Tale of Two Cities - Charles Dickens (Historical Fiction)
10. The Alchemist - Paulo Coelho (Philosophical Fiction)

## Structure

Each dataset file contains:

```json
{
  "dataset_id": "unique_identifier",
  "name": "Human-readable name",
  "description": "Purpose and scope",
  "version": "Semantic version",
  "created_at": "ISO date",
  "queries": [
    {
      "id": "query_###",
      "book": {
        "title": "Book Title",
        "author": "Author Name",
        "genre": "Genre",
        "year": 1900
      },
      "query": "User's travel request",
      "user_preferences": {
        "budget": "budget|moderate|luxury",
        "pace": "relaxed|moderate|fast",
        "duration_days": 5,
        "interests": ["list", "of", "interests"],
        "accessibility": "optional accessibility needs",
        "family": "optional family details"
      },
      "expected_output": {
        "locations": [
          {
            "city": "City Name",
            "country": "Country",
            "description": "Why this location matters",
            "must_visit": ["Landmark 1", "Landmark 2"]
          }
        ],
        "themes": ["Theme 1", "Theme 2"],
        "duration": "X-Y days"
      },
      "quality_criteria": {
        "book_relevance": "What to check",
        "preference_adherence": "What to verify",
        "completeness": "Expected coverage",
        "actionability": "Practical details needed",
        "geographical_accuracy": "Location requirements",
        "engagement": "Tone and style expectations"
      }
    }
  ],
  "scoring_functions": [
    {
      "name": "metric_name",
      "description": "What it measures",
      "scale": "1-5",
      "criteria": {
        "5": "Excellent",
        "4": "Good",
        "3": "Adequate",
        "2": "Poor",
        "1": "Failing"
      }
    }
  ]
}
```

## Usage

### Manual Evaluation

Use the expected outputs and quality criteria to manually evaluate system responses:

1. Run StoryLand AI with a query from the dataset
2. Compare the output against `expected_output`
3. Score each quality metric (1-5) using `quality_criteria`
4. Document any gaps or issues

### Automated Evaluation

The dataset can be integrated with evaluation tools:

```python
import json

# Load dataset
with open('eval/datasets/books_v1.json', 'r') as f:
    dataset = json.load(f)

# Run evaluations
for query in dataset['queries']:
    # Run StoryLand AI
    response = run_storyland(query['query'], query['user_preferences'])

    # Compare against expected output
    score = evaluate_response(response, query['expected_output'])

    # Log results
    print(f"Query {query['id']}: Score {score}")
```

### Integration with Langfuse

To create a Langfuse dataset from this file:

```python
from tools.langfuse_eval import LangfuseEvalPipeline

pipeline = LangfuseEvalPipeline()

# Convert to Langfuse dataset format
# (Implementation needed)
```

## Quality Metrics

The dataset includes 6 quality scoring functions:

1. **book_relevance** (1-5): Connection to book's settings, themes, or author
2. **preference_adherence** (1-5): Respect for user preferences
3. **completeness** (1-5): Coverage of cities, landmarks, author sites
4. **actionability** (1-5): Practical details and logistics
5. **geographical_accuracy** (1-5): Accuracy of locations
6. **engagement** (1-5): Writing quality and literary connection

### Interpreting Scores

- **5 (Excellent)**: Exceeds expectations, gold standard
- **4 (Good)**: Meets expectations with minor gaps
- **3 (Adequate)**: Acceptable but needs improvement
- **2 (Poor)**: Significant issues, major gaps
- **1 (Failing)**: Does not meet basic requirements

## Adding New Datasets

When creating new evaluation datasets:

1. Follow the JSON structure above
2. Use semantic versioning (e.g., `books_v2.json`)
3. Include diverse scenarios (genres, preferences, geographies)
4. Provide detailed expected outputs
5. Define clear quality criteria
6. Document the dataset in this README

## Best Practices

### Dataset Design

- **Diversity**: Cover edge cases and typical scenarios
- **Clarity**: Write clear, unambiguous queries
- **Realism**: Use realistic user preferences
- **Completeness**: Provide detailed expected outputs

### Expected Outputs

- **Specific**: Name actual cities, landmarks, museums
- **Accurate**: Verify all locations are real and correctly described
- **Comprehensive**: Include 2-4 locations per query
- **Contextual**: Explain why each location matters

### Quality Criteria

- **Measurable**: Define what "good" looks like for each metric
- **Specific**: Reference particular elements to check
- **Consistent**: Apply same standards across all queries
- **Relevant**: Focus on what matters for user experience

## Maintenance

### Regular Review

- Quarterly review of dataset quality
- Update expected outputs if system behavior changes intentionally
- Add new edge cases discovered through usage
- Archive outdated datasets

### Version History

- **v1.0** (2026-02-11): Initial dataset with 10 book queries

## Related Documentation

- [Langfuse Evaluation Pipeline](../../docs/langfuse-evaluation.md)
- [Testing Guide](../../docs/testing.md)
- [Evaluation Results](../README.md)

## Questions?

For issues with the dataset or suggestions for improvements, please open a GitHub issue.
