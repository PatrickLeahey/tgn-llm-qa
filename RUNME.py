# Databricks notebook source
# MAGIC %md
# MAGIC Before setting up the rest of the accelerator, we need set up a few credentials. Grab a personal access token for your Huggingface account ([documentation](https://huggingface.co/settings/token)). Here we demonstrate using the [Databricks Secret Scope](https://docs.databricks.com/security/secrets/secret-scopes.html) for credential management. 
# MAGIC
# MAGIC Populate values in cell 3 and run notebook
# MAGIC

# COMMAND ----------

import requests

# COMMAND ----------

databricks_host = ""
databricks_token = ""
huggingface_token = ""

# COMMAND ----------

endpoint = f"https://{databricks_host}/api/2.0/secrets/scopes/create"
data = {
  "scope": "tgn-llm-qa",
  "initial_manage_principal": "users",
  "scope_backend_type": "DATABRICKS"
}
headers = {"Authorization": f"Bearer {databricks_token}"}
requests.post(endpoint, json=data, headers=headers).text

# COMMAND ----------

endpoint = f"https://{databricks_host}/api/2.0/secrets/put"
data = {
  "scope": "tgn-llm-qa",
  "key": "huggingface",
  "string_value": {huggingface_token}
}
headers = {"Authorization": f"Bearer {databricks_token}"}
requests.post(endpoint, json=data, headers=headers)
