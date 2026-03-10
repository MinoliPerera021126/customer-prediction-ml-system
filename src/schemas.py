from pydantic import BaseModel


class CustomerInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    Credit: str
    balance: float
    housing_loan: str
    personal_loan: str
    contact: str
    last_contact_day: int
    last_contact_month: str
    last_contact_duration_sec: str
    campaign: int
    pdays: int
    previous: int
    previous_marketing_campaign: str