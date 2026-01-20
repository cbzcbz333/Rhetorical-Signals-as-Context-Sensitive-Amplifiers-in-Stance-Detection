from feature_extractor import OptimizedFeatureExtractor

extractor = OptimizedFeatureExtractor(cache_enabled=True)

print(extractor.extract_features("Isn't it obvious?", "rhetorical"))
print(extractor.extract_features("It might be true, but we are not sure.", "modality"))
