## Dataset Description

### Data Format
The dataset consists of diagnostically labelled images with additional metadata. The images are in JPEG format. The associated `.csv` file contains:
- A binary diagnostic label (`target`)
- Potential input variables (e.g., `age_approx`, `sex`, `anatom_site_general`, etc.)
- Additional attributes (e.g., image source and precise diagnosis)

### Prediction Task
In this challenge, you are tasked with differentiating benign from malignant cases. For each image (`isic_id`), you will assign a probability (`target`) ranging from [0, 1] that the case is malignant.

### The SLICE-3D Dataset
The dataset comprises skin lesion image crops extracted from 3D Total Body Photography (TBP) for skin cancer detection. To mimic non-dermoscopic images, this competition uses standardized cropped lesion-images from 3D TBP.

#### Vectra WB360
Vectra WB360, a 3D TBP product from Canfield Scientific, captures the complete visible cutaneous surface area in one macro-quality resolution tomographic image. AI-based software identifies individual lesions in a given 3D capture, enabling the image capture and identification of all lesions on a patient. These are exported as individual 15x15 mm field-of-view cropped photos.

#### Dataset Composition
The dataset contains every lesion from a subset of thousands of patients seen between 2015 and 2024 across nine institutions and three continents.

### Training Set Examples
- **Strongly-labelled tiles**: Labels derived through histopathology assessment.
- **Weak-labelled tiles**: Tiles not biopsied and considered 'benign' by a doctor.
