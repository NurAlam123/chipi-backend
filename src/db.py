import psycopg2
from typing import List, Dict, Tuple
import uuid


class ConversationDB:
    def __init__(self, db_config: Dict):
        """Initialize database connection."""
        self.db_config = db_config
        self.connection = None
        self.connect()

    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = False
            self._initialize_tables()
        except psycopg2.Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            raise

    def _initialize_tables(self):
        """Ensure required tables exist."""
        if not self.connection:
            return

        with self.connection.cursor() as cursor:
            try:
                # Create conversations table if not exists
                cursor.execute(
                    """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id UUID PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
                )

                # Create conversation_details table if not exists
                cursor.execute(
                    """
                CREATE TABLE IF NOT EXISTS conversation_details (
                    id SERIAL PRIMARY KEY,
                    session_id UUID REFERENCES conversations(session_id) ON DELETE CASCADE,
                    role VARCHAR(20) CHECK (role IN ('user', 'assistant', 'system')) NOT NULL,
                    content TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
                )

                # Create index if not exists
                cursor.execute(
                    """
                CREATE INDEX IF NOT EXISTS idx_session_id ON conversation_details(session_id)
                """
                )

                # Create trigger for updated_at if not exists
                cursor.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_conversations_updated_at') THEN
                            CREATE OR REPLACE FUNCTION update_updated_at()
                            RETURNS TRIGGER AS $_$
                            BEGIN
                                NEW.updated_at = CURRENT_TIMESTAMP;
                                RETURN NEW;
                            END;
                            $_$ LANGUAGE plpgsql;

                            CREATE TRIGGER update_conversations_updated_at
                            BEFORE UPDATE ON conversations
                            FOR EACH ROW
                            EXECUTE FUNCTION update_updated_at();
                        END IF;
                    END
                    $$;
                    """
                )
                self.connection.commit()
            except psycopg2.Error as e:
                self.connection.rollback()
                print(f"Error initializing tables: {e}")
                raise

    def create_conversation(self, initial_prompt: str | None = None) -> str:
        """Create a new conversation session."""
        if not initial_prompt:
            initial_prompt = "New Conversation"
        session_id = str(uuid.uuid4())
        title = (
            initial_prompt[:100] + "..."
            if initial_prompt and len(initial_prompt) > 100
            else initial_prompt
        )

        if not self.connection:
            return ""

        with self.connection.cursor() as cursor:
            try:
                cursor.execute(
                    "INSERT INTO conversations (session_id, title) VALUES (%s, %s)",
                    (session_id, title),
                )
                self.connection.commit()
                return session_id
            except psycopg2.Error as e:
                self.connection.rollback()
                print(f"Error creating conversation: {e}")
                raise

    def update_conversation_title(self, session_id: str, title: str) -> bool:
        """Update the title of a conversation."""
        if len(title) > 255:
            title = title[:252] + "..."

        if not self.connection:
            return False

        with self.connection.cursor() as cursor:
            try:
                cursor.execute(
                    "UPDATE conversations SET title = %s WHERE session_id = %s",
                    (title, session_id),
                )
                self.connection.commit()
                return cursor.rowcount > 0
            except psycopg2.Error as e:
                self.connection.rollback()
                print(f"Error updating conversation title: {e}")
                return False

    def add_message(
        self, session_id: str, role: str, content: str, reasoning: str | None
    ) -> bool:
        """Add a message to a conversation."""
        if not self.connection:
            return False

        if not reasoning:
            reasoning = ""

        with self.connection.cursor() as cursor:
            try:
                cursor.execute(
                    "INSERT INTO conversation_details (session_id, role, content, reasoning) VALUES (%s, %s, %s, %s)",
                    (session_id, role, content, reasoning),
                )

                # Update conversation title if it's the first message
                if role == "user":
                    cursor.execute(
                        "SELECT COUNT(*) FROM conversation_details WHERE session_id = %s",
                        (session_id,),
                    )
                    count = cursor.fetchone()[0]
                    if count == 1:  # First message
                        title = content[:100] + "..." if len(content) > 100 else content
                        self.update_conversation_title(session_id, title)

                self.connection.commit()
                return True
            except psycopg2.Error as e:
                self.connection.rollback()
                print(f"Error adding message: {e}")
                return False

    def get_conversation(self, session_id: str) -> Tuple[Dict, List[Dict]] | None:
        """Get a conversation with all its messages."""

        if not self.connection:
            return None

        with self.connection.cursor() as cursor:
            try:
                # Get conversation metadata
                cursor.execute(
                    "SELECT * FROM conversations WHERE session_id = %s", (session_id,)
                )
                conversation = cursor.fetchone()

                if not conversation:
                    return None

                # Convert to dict with column names
                columns = [desc[0] for desc in cursor.description]
                conversation_dict = dict(zip(columns, conversation))

                # Get conversation messages
                cursor.execute(
                    "SELECT role, content, timestamp, reasoning FROM conversation_details WHERE session_id = %s ORDER BY timestamp ASC",
                    (session_id,),
                )
                messages = cursor.fetchall()

                # Convert messages to list of dicts
                message_columns = [desc[0] for desc in cursor.description]
                messages_list = [dict(zip(message_columns, msg)) for msg in messages]

                return conversation_dict, messages_list
            except psycopg2.Error as e:
                print(f"Error getting conversation: {e}")
                return None

    def list_conversations(self, limit: int = 20, offset: int = 0) -> List[Dict]:
        """List all conversations with basic info."""
        if not self.connection:
            return [{}]
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(
                    "SELECT session_id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                    (limit, offset),
                )
                conversations = cursor.fetchall()

                # Convert to list of dicts with column names
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, conv)) for conv in conversations]
            except psycopg2.Error as e:
                print(f"Error listing conversations: {e}")
                return []

    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation and all its messages."""
        if not self.connection:
            return False
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(
                    "DELETE FROM conversations WHERE session_id = %s", (session_id,)
                )
                self.connection.commit()
                return cursor.rowcount > 0
            except psycopg2.Error as e:
                self.connection.rollback()
                print(f"Error deleting conversation: {e}")
                return False

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
