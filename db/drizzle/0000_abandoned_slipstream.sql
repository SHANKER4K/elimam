CREATE TABLE "conversations" (
	"session_id" text PRIMARY KEY NOT NULL,
	"messages" jsonb DEFAULT '[]'::jsonb NOT NULL
);
