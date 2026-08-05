"""On-disk `georef.db` reader — the local stand-in for the production ORM.

In production, keyframe poses, the local origin and level geometry come from
Postgres (`api_geokeyframe`, `GeoRef`, `GeoLevel`) via the Django ORM. Here they
come from the `georef.db` SQLite file that ships inside a map directory, so the
pipeline can be run against a plain checkout of map data with no Django install.
"""
