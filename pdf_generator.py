from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# --- PDF 1 Content (copy-pasted from above) ---
pdf1_data = [
    {
        "disease_name": "Malaria",
        "description": "Malaria is a serious and sometimes fatal disease caused by a parasite that commonly infects a certain type of mosquito which feeds on humans. People who get malaria are typically very sick with high fevers, shaking chills, and flu-like illness.",
        "icd_code": "B54"
    },
    {
        "disease_name": "Tuberculosis",
        "description": "Tuberculosis (TB) is a potentially serious infectious disease that mainly affects your lungs. The bacteria that cause tuberculosis are spread from person to person through tiny droplets released into the air via coughs and sneezes.",
        "icd_code": "A15.0"
    },
    {
        "disease_name": "Pneumonia",
        "description": "Pneumonia is an infection that inflames air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing.",
        "icd_code": "J18.9"
    },
    {
        "disease_name": "Diabetes Mellitus Type 2",
        "description": "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). With type 2 diabetes, your body either doesn't produce enough insulin, or it resists insulin.",
        "icd_code": "E11.9"
    },
    {
        "disease_name": "Hypertension",
        "description": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
        "icd_code": "I10"
    },
    {
        "disease_name": "Asthma",
        "description": "Asthma is a chronic lung disease that inflames and narrows the airways. Asthma causes recurring periods of wheezing (a whistling sound when you breathe), chest tightness, shortness of breath, and coughing.",
        "icd_code": "J45.909"
    },
    {
        "disease_name": "Alzheimer's Disease",
        "description": "Alzheimer's disease is a progressive neurological disorder that causes the brain to shrink and brain cells to die. It's the most common cause of dementia — a continuous decline in thinking, behavioral and social skills that affects a person's ability to function independently.",
        "icd_code": "G30.9"
    },
    {
        "disease_name": "Parkinson's Disease",
        "description": "Parkinson's disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves. Symptoms start gradually. The first symptom may be a barely noticeable tremor in just one limb.",
        "icd_code": "G20"
    },
    {
        "disease_name": "Osteoarthritis",
        "description": "Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage that cushions the ends of your bones wears down over time.",
        "icd_code": "M19.90"
    },
    {
        "disease_name": "Migraine",
        "description": "A migraine is a headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound.",
        "icd_code": "G43.909"
    }
]

# --- PDF 2 Content ---
pdf2_data = [
    {
        "disease_name": "Influenza (Flu)",
        "description": "Influenza is a contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs. It can cause mild to severe illness, and at times can lead to death.",
        "icd_code": "J11.1"
    },
    {
        "disease_name": "Common Cold",
        "description": "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold.",
        "icd_code": "J00"
    },
    {
        "disease_name": "Bronchitis",
        "description": "Bronchitis is an inflammation of the lining of your bronchial tubes, which carry air to and from your lungs. People who have bronchitis often cough up thickened mucus, which can be discolored.",
        "icd_code": "J40"
    },
    {
        "disease_name": "Gastroenteritis",
        "description": "Gastroenteritis is an inflammation of the stomach and intestines, typically resulting from bacterial toxins or viral infection and causing vomiting and diarrhea.",
        "icd_code": "A09"
    },
    {
        "disease_name": "Urinary Tract Infection (UTI)",
        "description": "A urinary tract infection (UTI) is an infection in any part of your urinary system — your kidneys, ureters, bladder and urethra. Most infections involve the lower urinary tract — the bladder and the urethra.",
        "icd_code": "N39.0"
    },
    {
        "disease_name": "Conjunctivitis",
        "description": "Conjunctivitis, also known as pinkeye, is an inflammation of the conjunctiva, the transparent membrane that lines your eyelid and covers the white part of your eyeball. When small blood vessels in the conjunctiva become inflamed, they're more visible, which is what causes the whites of your eyes to appear reddish or pinkish.",
        "icd_code": "H10.9"
    },
    {
        "disease_name": "Dermatitis",
        "description": "Dermatitis is a general term that describes a common skin irritation. It has many causes and forms and usually involves itchy, dry skin or a rash on swollen, reddened skin. Or it may cause the skin to blister, ooze, crust or flake.",
        "icd_code": "L30.9"
    },
    {
        "disease_name": "Anemia",
        "description": "Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. Having anemia can make you feel tired and weak.",
        "icd_code": "D64.9"
    },
    {
        "disease_name": "Hypothyroidism",
        "description": "Hypothyroidism (underactive thyroid) is a condition in which your thyroid gland doesn't produce enough of certain crucial hormones. Hypothyroidism may not cause noticeable symptoms in the early stages.",
        "icd_code": "E03.9"
    },
    {
        "disease_name": "Appendicitis",
        "description": "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. Appendicitis causes pain in your lower right abdomen.",
        "icd_code": "K37"
    }
]

# --- PDF 3 Content ---
pdf3_data = [
    {
        "disease_name": "Gout",
        "description": "Gout is a common and complex form of arthritis that can affect anyone. It's characterized by sudden, severe attacks of pain, swelling, redness and tenderness in one or more joints, most often in the big toe.",
        "icd_code": "M10.9"
    },
    {
        "disease_name": "Kidney Stones",
        "description": "Kidney stones (also called renal calculi, nephrolithiasis or urolithiasis) are hard deposits made of minerals and salt that form inside your kidneys. Kidney stones can affect any part of your urinary tract.",
        "icd_code": "N20.0"
    },
    {
        "disease_name": "Gallstones",
        "description": "Gallstones are hardened deposits of digestive fluid that can form in your gallbladder. Your gallbladder is a small, pear-shaped organ on the right side of your abdomen, just beneath your liver.",
        "icd_code": "K80.20"
    },
    {
        "disease_name": "Osteoporosis",
        "description": "Osteoporosis causes bones to become weak and brittle — so brittle that a fall or even mild stresses such as bending over or coughing can cause a fracture. Osteoporosis-related fractures most commonly occur in the hip, wrist or spine.",
        "icd_code": "M81.0"
    },
    {
        "disease_name": "Cataract",
        "description": "A cataract is a clouding of the normally clear lens of your eye. For people who have cataracts, seeing through cloudy lenses is a bit like looking through a frosty or fogged-up window.",
        "icd_code": "H26.9"
    },
    {
        "disease_name": "Glaucoma",
        "description": "Glaucoma is a group of eye conditions that damage the optic nerve, the health of which is vital for good vision. This damage is often caused by an abnormally high pressure in your eye.",
        "icd_code": "H40.90"
    },
    {
        "disease_name": "Depression",
        "description": "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. Also called major depressive disorder or clinical depression, it affects how you feel, think and behave and can lead to a variety of emotional and physical problems.",
        "icd_code": "F32.9"
    },
    {
        "disease_name": "Anxiety Disorder",
        "description": "Anxiety disorders are a group of mental illnesses that cause constant and overwhelming anxiety and fear. The excessive anxiety can lead to physical symptoms, such as a racing heart and shakiness.",
        "icd_code": "F41.9"
    },
    {
        "disease_name": "Insomnia",
        "description": "Insomnia is a common sleep disorder that can make it hard to fall asleep, hard to stay asleep, or cause you to wake up too early and not be able to get back to sleep. You may still feel tired when you wake up.",
        "icd_code": "G47.00"
    },
    {
        "disease_name": "Obesity",
        "description": "Obesity is a complex disease involving an excessive amount of body fat. Obesity isn't just a cosmetic concern. It's a medical problem that increases your risk of other diseases and health problems, such as heart disease, diabetes, high blood pressure and certain cancers.",
        "icd_code": "E66.9"
    }
]


def create_disease_pdf(filename, data):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<h2>Disease Information - {filename.replace('.pdf', '').replace('_', ' ').title()}</h2>", styles['h2']))
    story.append(Spacer(1, 0.2 * inch))

    for entry in data:
        story.append(Paragraph(f"<b>Disease Name:</b> {entry['disease_name']}", styles['h3']))
        story.append(Paragraph(f"<b>Description:</b> {entry['description']}", styles['Normal']))
        story.append(Paragraph(f"<b>ICD Code:</b> {entry['icd_code']}", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch)) # Add some space between entries

    doc.build(story)
    print(f"Created {filename}")

# Create the three PDF files
create_disease_pdf("document1.pdf", pdf1_data)
create_disease_pdf("document2.pdf", pdf2_data)
create_disease_pdf("document3.pdf", pdf3_data)