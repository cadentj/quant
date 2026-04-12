# %%


"""Pulls rubric results and run metadata into pandas via the Docent SDK."""

# README
# - Requires `pip install docent-python pandas`
# - API key is prefilled; rotate in Docent if you regenerate keys.
# - Run directly with `python <this file>`.

from __future__ import annotations

from docent import Docent

import os
from dotenv import load_dotenv
load_dotenv()

DOCENT_API_KEY = os.getenv("DOCENT_API_KEY")
SERVER_URL = "https://api.docent.transluce.org"
COLLECTION_ID = "2710c6a7-c1dc-4d6f-bd36-b74efcd3c6b9"

DQL_QUERY = """SELECT
  jr.id AS judge_result_id,
  jr.agent_run_id,
  jr.rubric_version,
  jr.result_type,
  jr.output->>'label' AS output_label,
  jr.output->>'explanation' AS output_explanation,
  jr.output AS output_json,
  jr.result_metadata,
  ar.created_at AS run_created_at,
  ar.metadata_json AS run_metadata
FROM judge_results jr
JOIN agent_runs ar ON ar.id = jr.agent_run_id
WHERE jr.rubric_id = 'd6d013a5-809c-4dfe-b8d6-146c684ba9ca' AND jr.rubric_version = 1
ORDER BY ar.created_at DESC"""


client = Docent(api_key=DOCENT_API_KEY, server_url=SERVER_URL)

result = client.execute_dql(COLLECTION_ID, DQL_QUERY)
df = client.dql_result_to_df_experimental(result)

# %%

