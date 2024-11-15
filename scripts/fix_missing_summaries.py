from asyncio import Semaphore, TaskGroup

from sqlalchemy import text

from code_helper.models import get_session, update_document_fragment_tsvector
from code_helper.index import summarize_code, with_retries


async def do_update(session, row, limiter):
    async with limiter:
        summary = await with_retries(summarize_code, row.fragment_content)
        await session.execute(
            text(
                """
                UPDATE document_fragments
                SET summary = :summary
                WHERE id = :id
            """
            ),
            {"id": row.id, "summary": summary},
        )
        await update_document_fragment_tsvector(session, row.id)


async def fix_missing_summaries(batch_size: int = 100, max_concurrency: int = 10):
    async with get_session() as session:
        limiter = Semaphore(max_concurrency)

        query = """
                    SELECT id, fragment_content
                    FROM document_fragments
                    WHERE summary IS NULL
                    limit 100
                """

        while True:
            result = await session.execute(text(query), {"batch_size": batch_size})
            rows = result.fetchall()

            if not rows:
                break

            print("Found", len(rows), " rows with missing fragment_content_tsv")

            async with limiter:
                for row in rows:
                    async with TaskGroup() as tg:
                        await tg.create_task(do_update(session, row, limiter))


if __name__ == "__main__":
    import asyncio

    asyncio.run(fix_missing_summaries())
