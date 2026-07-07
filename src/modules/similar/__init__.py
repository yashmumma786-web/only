"""
Similar Stone Explorer module.

Public surface: src.modules.similar.services
Internal logic: src.modules.similar.explorer
Data layer:     src.modules.similar.repository

Dependency rules:
  ✅ May call taxonomy.services
  ✅ May call ml_inference.services
  ✅ Owns data/similar_explorer.db (via repository.py)
  ❌ Must NOT import from search/
  ❌ Must NOT touch search's stone cache directly
"""
