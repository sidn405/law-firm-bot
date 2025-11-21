from datetime import datetime, timedelta
from simple_salesforce import Salesforce
from salesforce.query_methods import in_criteria
from typing import List, Dict, Any
from logging import Logger
import requests
import json
import os

log = Logger("salesforce")


class Connector:
    def __init__(
        self,
        username: str,
        password: str,
        security_token: str,
        subdomain: str = None,
        version: str = "44.0",
        max_retries=None,
        client_id=None,
    ):
        self.version = version
        self.username = username
        self.password = password
        self.security_token = security_token
        self.subdomain = subdomain
        self.max_retries = max_retries
        self.client_id = client_id
        self._new_session()
        self.base_url = (
            f"https://{self.session.sf_instance}/services/data/v{self.version}/"
        )

    def _new_session(self) -> Salesforce:
        if self.max_retries:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=self.max_retries)
            session.mount("https://", adapter)
        else:
            session = None
        self.session = Salesforce(
            username=self.username,
            password=self.password,
            security_token=self.security_token,
            domain=self.subdomain,
            session=session,
            client_id=self.client_id,
            version=self.version,
        )

    def call(self, method: str, url: str, data: Any = None) -> Any:
        payload = json.dumps(data) if data else None
        for e in [method, url, data]:
            log.debug(e)
        result = None
        try:
            result = self.session._call_salesforce(method, url, data=payload)
        except:
            if result and result.status_code == 401:
                self._new_session()
                result = self.session._call_salesforce(method, url, body=payload)
            else:
                raise
        return result.json() if result.text else None

    def add_attributes(
        self, data: dict, object_name: str, ref_id: str = None, other_attributes: dict = None
    ) -> dict:
        ta = {"type": object_name}
        ra = {**ta, **{"referenceId": ref_id}} if ref_id else ta
        attr = {**ra, **other_attributes} if other_attributes else ra
        return {**data, **{"attributes": attr}}

    def _chunk(self, records: List[Any], size: int):
        size = 200
        return [records[i : i + size] for i in range(0, len(records), size)]

    def _bulk_url(self):
        return f"{self.base_url}composite/sobjects"

    def _bulk_change(
        self,
        data: List[dict],
        method: str,
        all_or_none: bool = True,
        object_name: str = None,
        batch_size: int = 200,
    ) -> List[dict]:
        if batch_size > 200:
            raise ValueError("Salesforce does not accept batches of over 200 records")
        recs = (
            [self.add_attributes(d, object_name) for d in data] if object_name else data
        )
        chunks = self._chunk(recs, batch_size)
        results = []
        url = self._bulk_url()
        for chunk in chunks:
            load = {"allOrNone": all_or_none, "records": chunk}
            result = self.call(method=method, url=url, data=load)
            log.debug(result)
            results.extend(result)
        return results

    def bulk_create(
        self,
        data: List[dict],
        all_or_none: bool = True,
        object_name: str = None,
        batch_size: int = 200,
    ):
        return self._bulk_change(data, "POST", all_or_none, object_name, batch_size)

    def bulk_update(
        self,
        data: List[dict],
        all_or_none: bool = True,
        object_name: str = None,
        batch_size: int = 200,
    ):
        return self._bulk_change(data, "PATCH", all_or_none, object_name, batch_size)

    def bulk_delete(
        self, record_ids: List[str], all_or_none=True, batch_size: int = 200
    ) -> None:
        results = []
        chunks = self._chunk(record_ids, batch_size)
        for chunk in chunks:
            ids = ",".join(chunk)
            aon = "true" if all_or_none else "false"
            url = f"{self._bulk_url()}?ids={ids}&allOrNone={aon}"
            result = self.call("DELETE", url)
            results.append(result)
        return results

    def build_nested(
        self, parent_data: dict, children: Dict[str, List[Dict]]
    ) -> dict:
        for object_name, child in children.items():
            fmt_child = {object_name: {"records": child}}
            parent_data.update(fmt_child)
        return parent_data

    def nested_insert(self, data: List[dict], object_name: str) -> List[dict]:
        url = f"{self.base_url}composite/tree/{object_name}/"
        fmt_data = {"records": data}
        return self.call(method="POST", url=url, data=fmt_data)

    def _sobject_url(self, object_name: str) -> str:
        return f"{self.base_url}sobjects/{object_name}/"

    def create(self, object_name: str, data: dict) -> str:
        url = self._sobject_url(object_name)
        result = self.call(method="POST", url=url, data=data)
        if result.get("success") == True:
            return result["id"]
        else:
            raise Exception(result["errors"])

    def _id_url(self, record_id: str, object_name: str) -> str:
        return self._sobject_url(object_name) + record_id

    def update(self, record_id: str, object_name: str, data: dict) -> None:
        url = self._id_url(record_id, object_name)
        self.call(method="PATCH", url=url, data=data)

    def upsert(self, object_name: str, ext_id_fname: str, data: dict) -> str:
        url = f"{self._sobject_url(object_name)}{ext_id_fname}/{data.pop(ext_id_fname)}"
        result = self.call(method="PATCH", url=url, data=data)
        if result.get("success") == True:
            return result["id"]
        else:
            raise Exception(result["errors"])

    def delete(self, record_id: str, object_name: str) -> None:
        url = self._id_url(record_id, object_name)
        self.call(method="DELETE", url=url)

    def query(
        self,
        object_name: str,
        fields: List[str],
        criteria: str = None,
        values: List[Any] = None,
        query_all: bool = False,
    ) -> List[dict]:
        flds_str = ",".join(fields)
        soql_base = f"SELECT {flds_str} FROM {object_name} "
        crit = in_criteria(criteria, values)
        soql = soql_base + crit if crit else soql_base
        log.debug(soql)
        if query_all:
            results = self.session.query_all(soql)
        else:
            results = self.session.query(soql)
        return json.loads(json.dumps(results["records"]))
