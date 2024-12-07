# TODO: Adapt to test the MCP server
# import pytest

# @pytest.mark.asyncio
# async def test_search_code(test_client, files):
#     spaceship_file, _, _, _, _ = files
    
#     response = await test_client.post(
#         "/v1/code_search", 
#         json={"query_text": "spaceship"}
#     )
#     assert response.status_code == 200
#     assert response.json()["results"]
#     all_filepaths = [result["filepath"] for result in response.json()["results"]]
#     assert spaceship_file in all_filepaths