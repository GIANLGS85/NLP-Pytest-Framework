class TestDeIdentification:
    PHI_PATTERNS = ["patient_id", "SSN", "Mr. John"]

    def test_phi_removal(self, clinical_pipeline):
        text = "Patient John Doe SSN 123-45-6789 has fever"
        result = clinical_pipeline.annotate(text)
        deid_text = result["masked_with_entities_1"]

        for pattern in self.PHI_PATTERNS:
            assert pattern not in deid_text, f"PHI '{pattern}' not removed"
