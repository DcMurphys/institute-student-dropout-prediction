"""Readable names, help texts and status styling shown to staff."""

from __future__ import annotations

from app.models.student_analytics import AGE_BAND

FIELD_LABELS = {
    "Marital_status": "Marital status",
    "Application_mode": "Entry route",
    "Application_order": "Course preference",
    "Course": "Course",
    "Daytime_evening_attendance": "Attendance",
    "Previous_qualification": "Previous qualification",
    "Previous_qualification_grade": "Previous qualification grade (0-200)",
    "Admission_grade": "Admission grade (0-200)",
    "Nationality": "Nationality",
    "Displaced": "Studies away from home",
    "Educational_special_needs": "Special educational needs",
    "Debtor": "Outstanding debt",
    "Tuition_fees_up_to_date": "Tuition fees up to date",
    "Gender": "Gender",
    "Scholarship_holder": "Scholarship holder",
    "Age_at_enrollment": "Age at enrollment",
    "International": "International student",
    AGE_BAND: "Age at enrollment",
    "Overall_approval_rate": "First-year approval rate",
    "Sem1_approval_rate": "Semester 1 approval rate",
    "Sem2_approval_rate": "Semester 2 approval rate",
    "Overall_average_grade": "First-year average grade (0-20)",
    "Grade_vs_prev_qual": "Grade change vs. previous qualification",
}

SEMESTER_FIELD_LABELS = {
    "credited": "Units credited",
    "enrolled": "Units enrolled",
    "evaluations": "Evaluations taken",
    "approved": "Units approved",
    "grade": "Average grade (0-20)",
    "without_evaluations": "Units without evaluation",
}

FIELD_HELP = {
    "Application_order": "Where this course ranked in the student's application.",
    "Displaced": "The student moved away from home to study.",
    "Tuition_fees_up_to_date": "Fee status at the end of the first year.",
    "credited": "Units recognised from previous studies.",
    "evaluations": "Number of assessments the student sat.",
    "without_evaluations": "Enrolled units the student never sat an assessment for.",
}

# Progress measures offered on the dashboard (notebook Insights 2, 9 and 13)
PROGRESS_MEASURES = {
    "Overall_approval_rate": "Share of enrolled units passed in the first year",
    "Sem1_approval_rate": "Share of semester-1 units passed",
    "Sem2_approval_rate": "Share of semester-2 units passed",
    "Overall_average_grade": "Average grade of approved units, weighted by units",
    "Admission_grade": "Grade at admission",
    "Age_at_enrollment": "Age when the student enrolled",
}
RATE_MEASURES = {"Overall_approval_rate", "Sem1_approval_rate", "Sem2_approval_rate"}

# Status is never shown by colour alone: label + icon + colour
STATUS_DISPLAY = {
    "Dropout": {"label": "At risk of dropout", "icon": "▲", "css": "dropout"},
    "Enrolled": {"label": "Likely still enrolled", "icon": "●", "css": "enrolled"},
    "Graduate": {"label": "On track to graduate", "icon": "✓", "css": "graduate"},
}


def field_label(column: str) -> str:
    """Readable name for a column, including semester columns."""
    if column in FIELD_LABELS:
        return FIELD_LABELS[column]
    for semester, prefix in (("Sem. 1", "1st_sem_"), ("Sem. 2", "2nd_sem_")):
        if prefix in column:
            suffix = column.split(prefix, 1)[1]
            return f"{semester}: {SEMESTER_FIELD_LABELS.get(suffix, suffix)}"
    return column.replace("_", " ")
