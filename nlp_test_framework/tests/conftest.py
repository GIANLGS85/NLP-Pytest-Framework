import pytest
import json
#from pyspark.sql import SparkSession
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

@pytest.fixture(scope="session")
def spark_session():
    """Creates Spark session for ALL tests - only once per pytest run"""
    spark = sparknlp.start()  # Starts Spark + Spark NLP with default settings
    yield spark
    spark.stop()  # Clean shutdown

@pytest.fixture(scope="session")
def clinical_pipeline(spark_session):
    """Loads clinical NER pipeline - cached for entire test session"""
    return PretrainedPipeline("ner_clinical", lang="en")

@pytest.fixture
def patient_notes():
    """Sample patient data for testing - fresh for each test"""
    # Simulates reading from JSON file
    return json.loads('[{"text": "55 y.o. male with fever and cough"}]')

# Bonus fixtures for advanced testing
@pytest.fixture(params=[
    "Patient John Doe has fever",
    "55 y.o. female chest pain post-vaccine"
])
def clinical_text(request):
    return request.param
