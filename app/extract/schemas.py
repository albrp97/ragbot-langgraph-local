from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Type

class PersonForm(BaseModel):
    given_name: Optional[str] = Field(None, description="First name")
    family_name: Optional[str] = Field(None, description="Last name / Surname")
    date_of_birth: Optional[str] = None
    document_id: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class CVEducation(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    gpa: Optional[str] = None
    coursework: Optional[List[str]] = None
    honors: Optional[List[str]] = None

class CVExperience(BaseModel):
    role: Optional[str] = None
    organization: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    location: Optional[str] = None
    bullets: Optional[List[str]] = None
    technologies: Optional[List[str]] = None

class CVProject(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = None
    links: Optional[List[str]] = None

# We use only this schema for CV extraction
class CVStandard(BaseModel):
    # meta
    file_name: Optional[str] = None
    pages: Optional[int] = None
    is_cv: Optional[bool] = None
    detector_score: Optional[float] = None

    # contact
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    websites: Optional[List[str]] = None
    social_handles: Optional[List[str]] = None

    # profile
    objective: Optional[str] = None
    summary: Optional[str] = None

    # sections
    education: Optional[List[CVEducation]] = None
    experience: Optional[List[CVExperience]] = None
    projects: Optional[List[CVProject]] = None

    # skills & extras
    skills: Optional[Dict[str, List[str]]] = None   # by_category
    flat_skills: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    awards: Optional[List[str]] = None
    publications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    activities: Optional[List[str]] = None

SCHEMAS: Dict[str, Type[BaseModel]] = {
    "person_form": PersonForm,
    "cv_standard": CVStandard,
}
