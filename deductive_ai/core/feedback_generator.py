# Naive implementation, could use an agent backed by LLM to do this
class FeedbackGenerator:
    def generate_feedback(self, errors, swrl_rule):
        base_msg = "Found issues in your rule:\n"
        for error in errors:
            if "Undefined class" in error:
                cls = error.split()[-1]
                suggestions = self.om.suggest_similar(cls)
                base_msg += f"- {error}. Did you mean: {', '.join(suggestions)}?\n"
        return base_msg