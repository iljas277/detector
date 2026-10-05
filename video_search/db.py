from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATA, prepare_data


class Base(DeclarativeBase):
    pass


def utcnow():
    return datetime.now(timezone.utc)


class Target(Base):
    __tablename__ = "targets"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, default="")
    mode = Column(String, nullable=False)
    model_prompt = Column(String)
    prompt_confirmed = Column(Boolean, default=False)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class TargetReference(Base):
    __tablename__ = "target_references"
    id = Column(Integer, primary_key=True)
    target_id = Column(Integer, ForeignKey("targets.id"), nullable=False)
    version = Column(Integer, nullable=False)
    original_path = Column(String, nullable=False)
    crop_path = Column(String, nullable=False)
    bbox = Column(JSON, nullable=False)
    video_id = Column(Integer, ForeignKey("video_assets.id"))
    time_s = Column(Float)
    negative = Column(Boolean, default=False)


class VideoSource(Base):
    __tablename__ = "video_sources"
    id = Column(Integer, primary_key=True)
    kind = Column(String, nullable=False)
    camera_id = Column(String)
    flight_id = Column(String)
    location = Column(JSON)
    telemetry = Column(JSON)


class VideoAsset(Base):
    __tablename__ = "video_assets"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("video_sources.id"), nullable=False)
    original_name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    sha256 = Column(String, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    duration_s = Column(Float, nullable=False)
    stream_start_s = Column(Float, default=0)
    start_utc = Column(String)
    preview_path = Column(String)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("video_assets.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("targets.id"), nullable=False)
    target_version = Column(Integer, nullable=False)
    mode = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    model_revision = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    target_snapshot = Column(JSON, nullable=False)
    cache_key = Column(String, nullable=False)
    status = Column(String, default="queued")
    progress = Column(Float, default=0)
    error = Column(String)
    selected_device = Column(String)
    cancel_requested = Column(Boolean, default=False)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))


class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    local_id = Column(Integer, nullable=False)
    target_id = Column(Integer, nullable=False)


class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    target_id = Column(Integer, nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.id"))
    time_s = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=False)
    detection_score = Column(Float)
    similarity_score = Column(Float)
    score_origin = Column(String, nullable=False)
    reference_id = Column(Integer)
    observed = Column(Boolean, default=True)


class Episode(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id"), nullable=False)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    video_id = Column(Integer, nullable=False)
    source_id = Column(Integer, nullable=False)
    target_id = Column(Integer, nullable=False)
    start_s = Column(Float, nullable=False)
    end_s = Column(Float, nullable=False)
    observed_start_s = Column(Float, nullable=False)
    observed_end_s = Column(Float, nullable=False)
    rank_score = Column(Float)
    rank_rule = Column(String, nullable=False)
    preview_path = Column(String)
    status = Column(String, default="unreviewed")
    start_utc = Column(String)
    end_utc = Column(String)
    location = Column(JSON)


class ReviewDecision(Base):
    __tablename__ = "review_decisions"
    id = Column(Integer, primary_key=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"), nullable=False)
    status = Column(String, nullable=False)
    note = Column(String)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Collection(Base):
    __tablename__ = "collections"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, default="")


class CollectionEpisode(Base):
    __tablename__ = "collection_episodes"
    collection_id = Column(Integer, ForeignKey("collections.id"), primary_key=True)
    episode_id = Column(Integer, ForeignKey("episodes.id"), primary_key=True)


prepare_data()
engine = create_engine(f"sqlite:///{DATA / 'archive.sqlite3'}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    Base.metadata.create_all(engine)


@contextmanager
def session():
    db = Session()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
