def register_blueprints(app) -> None:
    from app.api.flashcard_routes import flashcards_bp
    from app.api.lecture_routes import lectures_bp
    from app.api.library_routes import library_bp
    from app.api.qa_routes import qa_bp
    from app.api.quiz_routes import quiz_bp
    from app.api.study_routes import study_bp

    for blueprint in (lectures_bp, library_bp, flashcards_bp, qa_bp, study_bp, quiz_bp):
        app.register_blueprint(blueprint)


__all__ = ["register_blueprints"]
