from sqlalchemy import text

from code_helper.models import get_session, update_document_fragment_tsvector


async def fix_missing_fragment_content_tsv():
    async with get_session() as session:
        query = """
                SELECT id, fragment_content
                FROM document_fragments
                WHERE fragment_content_tsv IS NULL
            """
        result = await session.execute(text(query))
        rows = result.fetchall()

        print("Found", len(rows), " rows with missing fragment_content_tsv")

        for row in rows:
            await update_document_fragment_tsvector(session, row.id)


if __name__ == "__main__":
    import asyncio

    asyncio.run(fix_missing_fragment_content_tsv())
