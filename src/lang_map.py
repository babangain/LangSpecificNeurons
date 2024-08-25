lang_map = {
    "set1": ["en", "fr", "es", "vi", "id", "ja", "zh"],
    "set2": ["en", "fr", "vi", "zh", "bn", "hi", "te", "mr"],
    "set3": ["bn", "hi", "te", "mr", "ur", "kn", "ml", "pa"],
    "set4": ["en", "fr", "es", "vi", "id", "ja", "zh", "bn", "hi", "te"],
    "set5": ["en", "fr", "es", "vi", "id", "ja", "zh", "bn", "hi", "te", "mr", "ur", "kn", "ml", "pa"]
}

lang_triplet_map = {
    "set1": [["en", "fr", "es"], ["fr", "vi", "zh"], ["vi", "ja", "zh"]],
    "set2": [["fr", "vi", "zh"], ["bn", "hi", "mr"], ["bn", "hi", "te"], ["fr", "bn", "te"]],
    "set3": [["bn", "hi", "mr"], ["bn", "hi", "ur"], ["te", "kn", "ml"], ["hi", "pa", "mr"], ["bn", "te", "pa"]],
    "set4": [["bn", "fr", "ja"], ["fr", "vi", "zh"], ["vi", "te", "es"], ["bn", "hi", "te"], ["en", "hi", "zh"]],
}