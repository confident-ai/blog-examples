from deepeval.metrics.factual_consistency import FactualConsistencyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.conceptual_similarity import ConceptualSimilarityMetric
from deepeval.metrics.llm_eval import LLMEvalMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test
import openai

def test_factual_correctness():
    input = "What if these shoes don't fit?"
    context = "All customers are eligible for a 30 day full refund at no extra costs."
    output = "We offer a 30-day full refund at no extra costs."
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context)
    assert_test(test_case, [factual_consistency_metric])

def test_relevancy():
    input = "What does your company do?"
    output = "Our company specializes in cloud computing"
    relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output)
    assert_test(test_case, [relevancy_metric])

def test_conceptual_similarity():
    input = "What did the cat do?"
    output = "The cat climbed up the tree"
    expected_output = "The cat ran up the tree."
    conceptual_similarity_metric = ConceptualSimilarityMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, expected_output=expected_output)
    assert_test(test_case, [conceptual_similarity_metric])

def test_humor():
    def make_chat_completion_request(prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    input = "Write me something funny related to programming"
    output = "Why did the programmer quit his job? Because he didn't get arrays!"
    llm_metric = LLMEvalMetric(
        criteria="How funny it is",
        completion_function=make_chat_completion_request
    )
    test_case = LLMTestCase(query=input, output=output)
    assert_test(test_case, [llm_metric])

def test_everything():
    input = "What did the cat do?"
    output = "The cat climbed up the tree"
    expected_output = "The cat ran up the tree."
    context = "The cat ran up the tree."
    conceptual_similarity_metric = ConceptualSimilarityMetric(minimum_score=0.5)
    relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
    factual_consistency_metric = FactualConsistencyMetric(minimum_score=0.5)
    test_case = LLMTestCase(query=input, output=output, context=context, expected_output=expected_output)
    assert_test(test_case, [conceptual_similarity_metric, relevancy_metric, factual_consistency_metric])