from pydantic import BaseModel, Base64UrlBytes, Base64UrlStr

class Base64InputEncodedModel(BaseModel):
    img_bytes: Base64UrlBytes

class Base64OutputEncodedModel(BaseModel):
    img_str: str