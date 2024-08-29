lang_map = {
    "set1": ["en", "fr", "es", "vi", "id", "ja", "zh"],
    "set2": ["en", "fr", "vi", "zh", "bn", "hi", "te", "mr"],
    "set3": ["bn", "hi", "ta", "te", "mr", "ur", "kn", "ml", "pa"],
    "set4": ["en", "fr", "es", "vi", "id", "ja", "zh", "bn", "hi", "te"],
    "set5": ["en", "fr", "es", "vi", "id", "ja", "zh", "bn", "hi", "ta", "te", "mr", "ur", "kn", "ml", "pa"]
}

lang_triplet_map = {
    "set1": [["en", "fr", "es"], ["fr", "vi", "zh"], ["vi", "ja", "zh"]],
    "set2": [["fr", "vi", "zh"], ["bn", "hi", "mr"], ["bn", "hi", "te"], ["fr", "bn", "te"]],
    "set3": [["bn", "hi", "mr"], ["bn", "hi", "ur"], ["te", "kn", "ml"], ["hi", "pa", "mr"], ["bn", "te", "pa"]],
    "set4": [["bn", "fr", "ja"], ["fr", "vi", "zh"], ["vi", "te", "es"], ["bn", "hi", "te"], ["en", "hi", "zh"]],
}

models_map = {
    "llama2": "meta-llama/Llama-2-7b-hf", # Done
    "llama2-ft": "meta-llama/Llama-2-7b-chat-hf", # Done
    "llama3": "meta-llama/Meta-Llama-3.1-8B", # Done
    "llama3-ft": "meta-llama/Meta-Llama-3.1-8B-Instruct", # Done
    "mistral": "mistralai/Mistral-7B-v0.3", # Done
    "mistral-ft": "mistralai/Mistral-7B-Instruct-v0.3", # Done
    "bloomz": "bigscience/bloomz-7b1", # Done
    "bloomz-mt": "bigscience/bloomz-7b1-mt", # Done
    "bloom": "bigscience/bloom-7b1", # Done
    "sarvam": "sarvamai/sarvam-2b-v0.5", # Done
    "aya101": "CohereForAI/aya-101",
    "aya23": "CohereForAI/aya-23-8B"
}

"""
Number of tokens seen in Million

Models      en      fr      es      vi      id      ja      zh      bn      hi      ta      te      mr      ur      kn      ml      pa
-------------------------------------------------------------------------------------------------------------------------------------------
llama2    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |
llama3    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |50     |100    |100    |100    |100    |
mistral   |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |
bloom     |100    |100    |100    |100    |100    |100    |100    |90     |70     |70     |70     |25     |60     |40     |50     |25     |
bloomz    |100    |100    |100    |100    |100    |100    |100    |90     |70     |70     |70     |25     |60     |40     |50     |25     |
bloomz-mt |100    |100    |100    |100    |100    |100    |100    |90     |70     |70     |70     |25     |60     |40     |50     |25     |
sarvam    |100    |100    |100    |100    |100    |100    |100    |100    |75     |90     |90     |30     |100    |50     |60     |30     |
aya23     |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |100    |70     |100    |100    |100    |100    |
aya101    |100    |100    |100    |100    |100    |100    |100    |100    |100    |80     |100    |40     |80     |55     |60     |40     |
    
"""