"""
Systematic Review Screening Results: Approaches for Assessing and Mitigating Algorithmic Bias in Health AI
Source: arXiv search
Screened: 28 papers
Included: 26 papers
Excluded: 2 papers

Exclusion reasons:
- Paper #6 (2407.12680): Uses AI to detect bias in medical curriculum text, not about bias IN health AI/ML algorithms
- Paper #19 (2504.07516): General AI fairness standards across industries, not specifically health-related methodology
"""

ARXIV_RESULTS = [
    # -------------------------------------------------------------------------
    # Paper 1
    # -------------------------------------------------------------------------
    {
        "title": "Detecting algorithmic bias in medical-AI models using trees",
        "authors": "Jeffrey Smith, Andre Holder, Rishikesan Kamaleswaran, Yao Xie",
        "year": 2023,
        "journal": "arXiv preprint",
        "arxiv_id": "2312.02959",
        "doi": None,
        "study_type": "Methodological development with empirical validation",
        "ai_ml_method": "Classification and Regression Trees (CART) with conformity scores",
        "health_domain": "Sepsis prediction / Critical care",
        "bias_axes": ["Race", "Gender", "Age"],
        "lifecycle_stage": "Post-deployment / Model evaluation",
        "assessment_or_mitigation": "Assessment",
        "approach_method": "Tree-based conformity score analysis for bias detection in clinical AI predictions; validated on synthetic data and real-world electronic medical records",
        "clinical_setting": "Hospital emergency / ICU (Georgia hospital)",
        "key_findings": "Demonstrated that CART-based conformity score methodology can detect areas where algorithmic bias emerges in sepsis prediction models; validated through controlled synthetic experiments and real-world EHR data from a Georgia hospital, showing practical applicability for clinical bias auditing."
    },
    # -------------------------------------------------------------------------
    # Paper 2
    # -------------------------------------------------------------------------
    {
        "title": "A Conceptual Algorithm for Applying Ethical Principles of AI to Medical Practice",
        "authors": "Debesh Jha, Gorkem Durak, Vanshali Sharma, Elif Keles, Vedat Cicek, Zheyuan Zhang, Abhishek Srivastava, Ashish Rauniyar, Desta Haileselassie Hagos, Nikhil Kumar Tomar, Frank H. Miller, Ahmet Topcu, Anis Yazidi, Jan Erik Haakegaard, Ulas Bagci",
        "year": 2023,
        "journal": "arXiv preprint",
        "arxiv_id": "2304.11530",
        "doi": None,
        "study_type": "Narrative review / Framework proposal",
        "ai_ml_method": "AI-enabled medical imaging systems (general)",
        "health_domain": "Medical imaging / General healthcare",
        "bias_axes": ["Race"],
        "lifecycle_stage": "Full lifecycle (data collection through deployment)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Conceptual algorithm for responsible AI implementation in healthcare; addresses racial bias in training datasets, limited model interpretability, and data scarcity",
        "clinical_setting": "Medical imaging diagnostics",
        "key_findings": "Proposed practical solutions for addressing key challenges in healthcare AI including data scarcity, racial bias in training datasets, and limited model interpretability; outlined a conceptual algorithm for responsible AI implementation in clinical practice."
    },
    # -------------------------------------------------------------------------
    # Paper 3
    # -------------------------------------------------------------------------
    {
        "title": "Algorithm Fairness in AI for Medicine and Healthcare",
        "authors": "Richard J. Chen, Tiffany Y. Chen, Jana Lipkova, Judy J. Wang, Drew F.K. Williamson, Ming Y. Lu, Sharifa Sahai, Faisal Mahmood",
        "year": 2021,
        "journal": "arXiv preprint",
        "arxiv_id": "2110.00603",
        "doi": None,
        "study_type": "Narrative review",
        "ai_ml_method": "General AI/ML including federated learning, disentanglement, model explainability",
        "health_domain": "General healthcare (diagnostics, treatment, billing)",
        "bias_axes": ["Race", "Gender", "Socioeconomic status"],
        "lifecycle_stage": "Full lifecycle (image acquisition, genetic variation, labeling, model development)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Review of sources of algorithmic bias within clinical workflows and emerging mitigation approaches including federated learning, disentanglement, and model explainability in AI as Software as a Medical Device (AI-SaMD)",
        "clinical_setting": "General clinical settings (diagnostics, treatment, billing)",
        "key_findings": "Identified sources of algorithmic bias in clinical workflows including image acquisition, genetic variation, and labeling inconsistencies; discussed emerging mitigation approaches including federated learning, disentanglement, and model explainability for AI-SaMD development."
    },
    # -------------------------------------------------------------------------
    # Paper 4
    # -------------------------------------------------------------------------
    {
        "title": "Algorithmic Bias, Generalist Models, and Clinical Medicine",
        "authors": "Geoff Keeling",
        "year": 2023,
        "journal": "arXiv preprint",
        "arxiv_id": "2305.04008",
        "doi": None,
        "study_type": "Conceptual analysis / Perspective",
        "ai_ml_method": "Generalist language models (BERT, PaLM) fine-tuned for clinical tasks",
        "health_domain": "General clinical medicine",
        "bias_axes": ["Multiple (general algorithmic bias)"],
        "lifecycle_stage": "Model development (pre-training, fine-tuning, prompting)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Analysis of how biases in generalist models differ from biases in prior narrow clinical models; provides practical mitigation strategies for LLM-based clinical systems",
        "clinical_setting": "General clinical decision support",
        "key_findings": "Explained how generalist models introduce distinct forms of algorithmic bias compared to traditional narrow clinical ML systems; demonstrated that the relationship between model biases and training data biases is more complex in generalist models; provided practical mitigation strategies."
    },
    # -------------------------------------------------------------------------
    # Paper 5
    # -------------------------------------------------------------------------
    {
        "title": "Exploring Bias and Prediction Metrics to Characterise the Fairness of Machine Learning for Equity-Centered Public Health Decision-Making: A Narrative Review",
        "authors": "Shaina Raza, Arjan Vreeken, Sennay Ghebreab, Hiroshi Mamiya",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2408.13295",
        "doi": None,
        "study_type": "Narrative review (systematic search)",
        "ai_ml_method": "General ML methods for public health surveillance and research",
        "health_domain": "Public health",
        "bias_axes": ["Multiple (equity lens including race, gender, socioeconomic)"],
        "lifecycle_stage": "Model evaluation / Assessment",
        "assessment_or_mitigation": "Assessment",
        "approach_method": "Systematic literature search (PubMed, IEEE, 2008-2023) analyzing 72 articles; characterized commonly observed bias types and corresponding quantitative fairness assessment methods evaluated through an equity lens",
        "clinical_setting": "Public health research and surveillance",
        "key_findings": "Analyzed 72 articles to document commonly observed bias types and quantitative assessment methods; proposed a formalized evaluation framework for ML in public health from an equity perspective."
    },
    # -------------------------------------------------------------------------
    # Paper 7
    # -------------------------------------------------------------------------
    {
        "title": "Bias by Design? How Data Practices Shape Fairness in AI Healthcare Systems",
        "authors": "Anna Arias-Duart, Maria Eugenia Cardello, Atia Cortes",
        "year": 2025,
        "journal": "arXiv preprint",
        "arxiv_id": "2510.20332",
        "doi": None,
        "study_type": "Case study / Empirical analysis",
        "ai_ml_method": "Clinical AI systems (general)",
        "health_domain": "Healthy aging / General clinical AI",
        "bias_axes": ["Sex", "Gender", "Age", "Habitat", "Socioeconomic status"],
        "lifecycle_stage": "Data collection and preprocessing",
        "assessment_or_mitigation": "Both",
        "approach_method": "Analysis of training data quality effects on fairness in clinical AI from Spain's AI4HealthyAging initiative; identified historical, representation, and measurement biases across multiple use cases",
        "clinical_setting": "Clinical AI (AI4HealthyAging initiative, Spain)",
        "key_findings": "Identified several types of bias across multiple use cases including historical, representation, and measurement biases across demographic factors; provided practical recommendations for improving fairness and robustness of clinical problem design and data collection."
    },
    # -------------------------------------------------------------------------
    # Paper 8
    # -------------------------------------------------------------------------
    {
        "title": "Machine Learning and Public Health: Identifying and Mitigating Algorithmic Bias through a Systematic Review",
        "authors": "Sara Altamirano, Arjan Vreeken, Sennay Ghebreab",
        "year": 2025,
        "journal": "arXiv preprint",
        "arxiv_id": "2510.14669",
        "doi": None,
        "study_type": "Systematic review",
        "ai_ml_method": "General ML methods for public health",
        "health_domain": "Public health (Netherlands)",
        "bias_axes": ["Multiple (subgroup-level disparities)"],
        "lifecycle_stage": "Full lifecycle (awareness through reporting)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Developed RABAT (bias assessment tool) combining established evaluation frameworks; introduced ACAR framework (Awareness, Conceptualization, Application, Reporting) for addressing fairness throughout ML development",
        "clinical_setting": "Dutch public health ML applications",
        "key_findings": "Analyzed 35 peer-reviewed studies (2021-2025); found significant gaps: while data sampling practices are well-documented, most studies lack explicit fairness considerations, subgroup analyses, and transparent discussion of potential harms; introduced RABAT assessment tool and ACAR four-stage framework."
    },
    # -------------------------------------------------------------------------
    # Paper 9
    # -------------------------------------------------------------------------
    {
        "title": "Artificial Intelligence-Driven Clinical Decision Support Systems",
        "authors": "Muhammet Alkan, Idris Zakariyya, Samuel Leighton, Kaushik Bhargav Sivangi, Christos Anagnostopoulos, Fani Deligianni",
        "year": 2025,
        "journal": "arXiv preprint",
        "arxiv_id": "2501.09628",
        "doi": None,
        "study_type": "Review / Survey",
        "ai_ml_method": "ML for clinical prediction models; differential privacy; federated learning",
        "health_domain": "Clinical decision support (general)",
        "bias_axes": ["Multiple (demographic disparities in clinical predictions)"],
        "lifecycle_stage": "Full lifecycle (development, validation, deployment)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Review of techniques for detecting and reducing bias in clinical prediction models; examines model calibration, decision curve analysis, explainability methods, privacy-preservation techniques (differential privacy, federated learning)",
        "clinical_setting": "General clinical decision support",
        "key_findings": "Emphasized that trustworthy healthcare AI requires addressing fairness, explainability, and privacy; discussed techniques for detecting and reducing bias in clinical prediction models; analyzed trade-offs between privacy safeguards and model performance."
    },
    # -------------------------------------------------------------------------
    # Paper 10
    # -------------------------------------------------------------------------
    {
        "title": "Identifying and mitigating bias in algorithms used to manage patients in a pandemic",
        "authors": "Yifan Li, Garrett Yoon, Mustafa Nasir-Moin, David Rosenberg, Sean Neifert, Douglas Kondziolka, Eric Karl Oermann",
        "year": 2021,
        "journal": "arXiv preprint",
        "arxiv_id": "2111.00340",
        "doi": None,
        "study_type": "Empirical study",
        "ai_ml_method": "Logistic regression with calibration techniques",
        "health_domain": "COVID-19 pandemic management (mortality, ventilator needs, hospitalization prediction)",
        "bias_axes": ["Race", "Gender", "Age"],
        "lifecycle_stage": "Model development (training-time calibration)",
        "assessment_or_mitigation": "Both (assessment via fairness metrics, mitigation via calibration)",
        "approach_method": "Developed logistic regression models for pandemic clinical outcomes; applied calibration techniques during training to create fairer models; evaluated on EHR data from four NYC hospitals",
        "clinical_setting": "Hospital (four New York City hospitals)",
        "key_findings": "Calibrated models reduced biased outcomes by 57% while maintaining predictive accuracy (AUC); average sensitivity improved from 0.527 to 0.955 post-calibration; demonstrated that simple calibration adjustments during model training can lead to substantial and sustained gains in fairness."
    },
    # -------------------------------------------------------------------------
    # Paper 11
    # -------------------------------------------------------------------------
    {
        "title": "Towards Clinical AI Fairness: Filling Gaps in the Puzzle",
        "authors": "Mingxuan Liu, Yilin Ning, Salinelat Teixayavong, Xiaoxuan Liu, Mayli Mertens, Yuqing Shang, Xin Li, Di Miao, Jie Xu, Daniel Shu Wei Ting, Lionel Tim-Ee Cheng, Jasmine Chiat Ling Ong, Zhen Ling Teo, Ting Fang Tan, Narrendar RaviChandran, Fei Wang, Leo Anthony Celi, Marcus Eng Hock Ong, Nan Liu",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2405.17921",
        "doi": None,
        "study_type": "Evidence gap analysis / Perspective",
        "ai_ml_method": "General AI/ML across clinical specialties",
        "health_domain": "Multiple clinical domains",
        "bias_axes": ["Multiple (group-level and individual-level)"],
        "lifecycle_stage": "Model evaluation and deployment",
        "assessment_or_mitigation": "Both",
        "approach_method": "Evidence gap analysis of AI fairness across medical specialties; identifies insufficient research in many clinical domains and overemphasis on group-level fairness metrics over individual equity",
        "clinical_setting": "Multiple clinical specialties",
        "key_findings": "Revealed two major shortcomings: insufficient research on AI fairness in many clinical domains where AI is deployed, and overemphasis on group-level fairness metrics rather than individual equity considerations; called for integrating healthcare professionals' expertise into fairness development."
    },
    # -------------------------------------------------------------------------
    # Paper 12
    # -------------------------------------------------------------------------
    {
        "title": "A Design Framework for operationalizing Trustworthy Artificial Intelligence in Healthcare: Requirements, Tradeoffs and Challenges for its Clinical Adoption",
        "authors": "Pedro A. Moreno-Sanchez, Javier Del Ser, Mark van Gils, Jussi Hernesniemi",
        "year": 2025,
        "journal": "arXiv preprint",
        "arxiv_id": "2504.19179",
        "doi": None,
        "study_type": "Framework proposal / Review",
        "ai_ml_method": "General clinical AI systems",
        "health_domain": "Cardiovascular disease / General healthcare",
        "bias_axes": ["Multiple (demographic disparities)"],
        "lifecycle_stage": "Full lifecycle (design through deployment)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Disease-agnostic framework integrating trustworthy AI principles (human oversight, robustness, privacy, transparency, bias mitigation, accountability) with requirements across diverse stakeholders; demonstrated on cardiovascular disease applications",
        "clinical_setting": "Cardiovascular disease diagnostics and general clinical settings",
        "key_findings": "Identified key principles for trustworthy healthcare AI including bias mitigation; outlined disease-agnostic requirements across diverse stakeholders (clinicians, patients, providers, regulators); examined practical tradeoffs in implementing trustworthy AI in clinical settings."
    },
    # -------------------------------------------------------------------------
    # Paper 13
    # -------------------------------------------------------------------------
    {
        "title": "Fair by design: A sociotechnical approach to justifying the fairness of AI-enabled systems across the lifecycle",
        "authors": "Marten H.L. Kaas, Christopher Burr, Zoe Porter, Berk Ozturk, Philippa Ryan, Michael Katell, Nuala Polo, Kalle Westerling, Ibrahim Habli",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2406.09029",
        "doi": None,
        "study_type": "Framework proposal with case study",
        "ai_ml_method": "Clinical diagnostic support system (hypertension risk prediction)",
        "health_domain": "Hypertension risk prediction in diabetic patients",
        "bias_axes": ["Multiple (patient population-level disparities)"],
        "lifecycle_stage": "Full lifecycle (all stages of AI system lifecycle)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Trustworthy and Ethical Assurance (TEA) framework with open-source tools for broadening fairness evaluation across all stages of AI lifecycle; demonstrated through clinical diagnostic support system for hypertension risk in diabetic patients",
        "clinical_setting": "Clinical diagnostics (hypertension risk in diabetic patients)",
        "key_findings": "Proposed TEA framework that broadens fairness evaluation beyond statistical measures to encompass all stages of AI system lifecycle; demonstrated approach through a clinical diagnostic support system; established replicable practices for achieving fairness in healthcare AI."
    },
    # -------------------------------------------------------------------------
    # Paper 14
    # -------------------------------------------------------------------------
    {
        "title": "Addressing Fairness Issues in Deep Learning-Based Medical Image Analysis: A Systematic Review",
        "authors": "Zikang Xu, Jun Li, Qingsong Yao, Han Li, Mingyue Zhao, S. Kevin Zhou",
        "year": 2022,
        "journal": "arXiv preprint",
        "arxiv_id": "2209.13177",
        "doi": None,
        "study_type": "Systematic review",
        "ai_ml_method": "Deep learning for medical image analysis",
        "health_domain": "Medical imaging (general)",
        "bias_axes": ["Age", "Sex/Gender", "Race/Ethnicity"],
        "lifecycle_stage": "Model development and evaluation",
        "assessment_or_mitigation": "Both",
        "approach_method": "Systematic review categorizing research into fairness evaluation and unfairness mitigation in deep learning-based medical imaging; identifies performance disparities across demographic subgroups",
        "clinical_setting": "Medical imaging diagnostics",
        "key_findings": "Categorized current research into fairness evaluation and unfairness mitigation; identified performance disparities across demographic subgroups in medical imaging (e.g., poorer predictive performance in elderly females); aimed to build shared understanding between AI researchers and clinicians about fairness challenges."
    },
    # -------------------------------------------------------------------------
    # Paper 15
    # -------------------------------------------------------------------------
    {
        "title": "Fairness-Aware Interpretable Modeling (FAIM) for Trustworthy Machine Learning in Healthcare",
        "authors": "Mingxuan Liu, Yilin Ning, Yuhe Ke, Yuqing Shang, Bibhas Chakraborty, Marcus Eng Hock Ong, Roger Vaughan, Nan Liu",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2403.05235",
        "doi": None,
        "study_type": "Methodological development with empirical validation",
        "ai_ml_method": "Interpretable ML models with fairness-aware optimization; interactive model selection interface",
        "health_domain": "Emergency department hospital admission prediction",
        "bias_axes": ["Sex", "Race"],
        "lifecycle_stage": "Model development (in-processing)",
        "assessment_or_mitigation": "Mitigation",
        "approach_method": "FAIM framework: interpretable modeling with interactive interface for selecting fairer models from high-performing options; facilitates collaboration between data-driven insights and clinical expertise; tested on MIMIC-IV-ED and SGH-ED databases",
        "clinical_setting": "Emergency department (MIMIC-IV-ED and SGH-ED databases)",
        "key_findings": "Successfully reduced sex and race biases in hospital admission predictions; models maintained strong discriminatory capability while significantly mitigating biases, outperforming commonly used bias-mitigation methods; demonstrated fairness improvements achievable without sacrificing model performance."
    },
    # -------------------------------------------------------------------------
    # Paper 16
    # -------------------------------------------------------------------------
    {
        "title": "MLHOps: Machine Learning for Healthcare Operations",
        "authors": "Faiza Khan Khattak, Vallijah Subasri, Amrit Krishnan, Elham Dolatabadi, Deval Pandya, Laleh Seyyed-Kalantari, Frank Rudzicz",
        "year": 2023,
        "journal": "arXiv preprint",
        "arxiv_id": "2305.02474",
        "doi": None,
        "study_type": "Review / Practical guidance",
        "ai_ml_method": "General ML operations pipeline for healthcare",
        "health_domain": "General healthcare operations",
        "bias_axes": ["Multiple (general fairness and equity)"],
        "lifecycle_stage": "Full lifecycle (conception through deployment and management)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Comprehensive MLOps framework for healthcare covering data sources, processing, feature engineering, performance monitoring, model refreshing, bias mitigation, fairness, transparency, and data protection",
        "clinical_setting": "General clinical ML deployment",
        "key_findings": "Defined MLHOps as combining processes for dependable, effective, and ethical ML implementation in healthcare; provided comprehensive guidance spanning the entire ML lifecycle including bias mitigation, equitable outcomes, model transparency, and data protection."
    },
    # -------------------------------------------------------------------------
    # Paper 17
    # -------------------------------------------------------------------------
    {
        "title": "Towards clinical AI fairness: A translational perspective",
        "authors": "Mingxuan Liu, Yilin Ning, Salinelat Teixayavong, Mayli Mertens, Jie Xu, Daniel Shu Wei Ting, Lionel Tim-Ee Cheng, Jasmine Chiat Ling Ong, Zhen Ling Teo, Ting Fang Tan, Ravi Chandran Narrendar, Fei Wang, Leo Anthony Celi, Marcus Eng Hock Ong, Nan Liu",
        "year": 2023,
        "journal": "arXiv preprint",
        "arxiv_id": "2304.13493",
        "doi": None,
        "study_type": "Perspective / Commentary",
        "ai_ml_method": "General clinical AI systems",
        "health_domain": "General clinical healthcare",
        "bias_axes": ["Multiple (general fairness concerns)"],
        "lifecycle_stage": "Translation and deployment",
        "assessment_or_mitigation": "Both",
        "approach_method": "Translational analysis identifying misalignment between technical and clinical conceptualizations of AI fairness; proposes actionable approaches for addressing fairness issues specific to healthcare contexts through interdisciplinary teams",
        "clinical_setting": "General clinical practice",
        "key_findings": "Identified misalignment between how technical experts and clinicians conceptualize AI fairness; outlined barriers preventing fair AI systems from translating into clinical practice; emphasized necessity for interdisciplinary teams to close knowledge gaps."
    },
    # -------------------------------------------------------------------------
    # Paper 18
    # -------------------------------------------------------------------------
    {
        "title": "A survey of recent methods for addressing AI fairness and bias in biomedicine",
        "authors": "Yifan Yang, Mingquan Lin, Han Zhao, Yifan Peng, Furong Huang, Zhiyong Lu",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2402.08250",
        "doi": None,
        "study_type": "Systematic survey",
        "ai_ml_method": "NLP and Computer Vision methods in biomedicine; debiasing techniques (distributional and algorithmic)",
        "health_domain": "Biomedicine (clinical diagnostics, surgical decision-making)",
        "bias_axes": ["Race", "Gender"],
        "lifecycle_stage": "Model development (before, during, and after model development)",
        "assessment_or_mitigation": "Mitigation",
        "approach_method": "Systematic survey of debiasing methods in biomedical NLP and CV; searched PubMed, ACM, IEEE Xplore (Jan 2018-Dec 2023); filtered 10,041 articles to 55 included; categorized methods as distributional or algorithmic",
        "clinical_setting": "Biomedical NLP and computer vision applications",
        "key_findings": "Surveyed 55 articles on debiasing methods in biomedical NLP and CV; categorized existing methods as distributional or algorithmic; identified that bias can originate from multiple sources before, during, or after model development; reviewed potential methods from general domain applicable to biomedicine."
    },
    # -------------------------------------------------------------------------
    # Paper 20
    # -------------------------------------------------------------------------
    {
        "title": "Evaluating Fair Feature Selection in Machine Learning for Healthcare",
        "authors": "Md Rahat Shahriar Zawad, Peter Washington",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2403.19165",
        "doi": None,
        "study_type": "Empirical study",
        "ai_ml_method": "Fair feature selection methods for ML classifiers",
        "health_domain": "General healthcare (multiple datasets)",
        "bias_axes": ["Demographic group disparities (multiple)"],
        "lifecycle_stage": "Data preprocessing / Feature selection",
        "assessment_or_mitigation": "Mitigation",
        "approach_method": "Fair feature selection methodology that considers equal importance to all demographic groups, balancing fairness metrics alongside error reduction; tested across three publicly accessible healthcare datasets",
        "clinical_setting": "General healthcare ML applications",
        "key_findings": "Demonstrated that fairness-aware feature selection improves fairness with only slight decreases in balanced accuracy; addressed both distributive and procedural dimensions of equitable ML; showed conventional feature selection fails to account for demographic population variations."
    },
    # -------------------------------------------------------------------------
    # Paper 21
    # -------------------------------------------------------------------------
    {
        "title": "How Can We Diagnose and Treat Bias in Large Language Models for Clinical Decision-Making?",
        "authors": "Kenza Benkirane, Jackie Kay, Maria Perez-Ortiz",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2410.16574",
        "doi": None,
        "study_type": "Empirical study with framework development",
        "ai_ml_method": "Large Language Models (8 LLMs tested with various prompting and fine-tuning strategies)",
        "health_domain": "Clinical decision-making (multiple medical fields)",
        "bias_axes": ["Gender", "Ethnicity"],
        "lifecycle_stage": "Model evaluation and post-processing",
        "assessment_or_mitigation": "Both",
        "approach_method": "Developed Counterfactual Patient Variations dataset derived from JAMA Clinical Challenge; evaluation framework using MCQs and explanations; tested 8 LLMs with various prompting strategies and fine-tuning approaches",
        "clinical_setting": "Clinical decision-making (JAMA Clinical Challenge scenarios)",
        "key_findings": "Found that mitigating one type of bias can inadvertently introduce another; gender bias differs substantially across medical fields; examining both correct answers and reasoning is essential since responses may be accurate despite biased logic; proposed methods for assessing bias in real clinical scenarios."
    },
    # -------------------------------------------------------------------------
    # Paper 22
    # -------------------------------------------------------------------------
    {
        "title": "Understanding-informed Bias Mitigation for Fair CMR Segmentation",
        "authors": "Tiarna Lee, Esther Puyol-Anton, Bram Ruijsink, Pier-Giorgio Masci, Louise Keehn, Phil Chowienczyk, Emily Haseler, Miaojing Shi, Andrew P. King",
        "year": 2025,
        "journal": "arXiv preprint",
        "arxiv_id": "2503.17089",
        "doi": None,
        "study_type": "Empirical study",
        "ai_ml_method": "Deep learning for cardiac MR segmentation; oversampling, importance reweighting, Group DRO",
        "health_domain": "Cardiac magnetic resonance (CMR) imaging / Cardiology",
        "bias_axes": ["Ethnicity (Black vs. White)"],
        "lifecycle_stage": "Model development (training-time mitigation)",
        "assessment_or_mitigation": "Mitigation",
        "approach_method": "Applied bias mitigation techniques including oversampling, importance reweighting, and Group DRO to cardiac MR segmentation models; evaluated on unbalanced datasets with external clinical validation",
        "clinical_setting": "Cardiac MR imaging",
        "key_findings": "Oversampling significantly improved performance for underrepresented Black subjects without significantly reducing majority White subjects' performance; cropped image formats improved outcomes for both groups; external clinical validation showed strong segmentation results with no statistically significant bias remaining."
    },
    # -------------------------------------------------------------------------
    # Paper 23
    # -------------------------------------------------------------------------
    {
        "title": "Unmasking Bias in AI: A Systematic Review of Bias Detection and Mitigation Strategies in Electronic Health Record-based Models",
        "authors": "Feng Chen, Liqin Wang, Julie Hong, Jiaqi Jiang, Li Zhou",
        "year": 2023,
        "journal": "arXiv preprint (published in JAMIA, May 2024)",
        "arxiv_id": "2310.19917",
        "doi": None,
        "study_type": "Systematic review",
        "ai_ml_method": "AI/ML models built on electronic health record data",
        "health_domain": "EHR-based predictive healthcare",
        "bias_axes": ["Algorithmic", "Confounding", "Implicit", "Measurement", "Selection", "Temporal"],
        "lifecycle_stage": "Full lifecycle (data collection, preprocessing, model development, evaluation)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Systematic review of 450 articles (2010-Dec 2023), selected 20 meeting criteria; identified six bias categories; examined detection methods using fairness metrics (statistical parity, equal opportunity, predictive equity) and mitigation via resampling, reweighting, transformation",
        "clinical_setting": "EHR-based clinical applications",
        "key_findings": "Identified six primary bias categories (algorithmic, confounding, implicit, measurement, selection, temporal); most AI applications focused on predictive tasks; mitigation approaches primarily addressed implicit and selection biases through data preprocessing; called for standardized, generalizable methodologies for handling bias in EHR-based models."
    },
    # -------------------------------------------------------------------------
    # Paper 24
    # -------------------------------------------------------------------------
    {
        "title": "Datasheets for Healthcare AI: A Framework for Transparency and Bias Mitigation",
        "authors": "Marjia Siddik, Harshvardhan J. Pandit",
        "year": 2025,
        "journal": "arXiv preprint",
        "arxiv_id": "2501.05617",
        "doi": None,
        "study_type": "Framework proposal",
        "ai_ml_method": "General healthcare AI systems (dataset documentation)",
        "health_domain": "General healthcare AI",
        "bias_axes": ["Multiple (data quality-related biases)"],
        "lifecycle_stage": "Data collection and documentation (pre-processing)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Healthcare AI Datasheet framework for enhanced dataset documentation; machine-readable format supporting automated risk assessment and regulatory compliance; identifies gaps in current documentation preventing bias recognition",
        "clinical_setting": "General clinical AI applications",
        "key_findings": "Identified gaps in current dataset documentation methods that prevent recognition and mitigation of bias; introduced Healthcare AI Datasheet framework enhancing transparency; framework supports machine-readable formatting for automated risk assessment and regulatory compliance."
    },
    # -------------------------------------------------------------------------
    # Paper 25
    # -------------------------------------------------------------------------
    {
        "title": "Fairness at Every Intersection: Uncovering and Mitigating Intersectional Biases in Multimodal Clinical Predictions",
        "authors": "Resmi Ramachandranpillai, Kishore Sampath, Ayaazuddin Mohammad, Malihe Alikhani",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2412.00606",
        "doi": None,
        "study_type": "Empirical study",
        "ai_ml_method": "Pre-trained clinical language models; multimodal data fusion (text, time series, tabular, events, images)",
        "health_domain": "Emergency department clinical predictions",
        "bias_axes": ["Race", "Gender", "Ethnicity", "Intersectional (overlapping characteristics)"],
        "lifecycle_stage": "Model development and evaluation",
        "assessment_or_mitigation": "Both",
        "approach_method": "Analysis of intersectional biases in multimodal EHR data using pre-trained clinical language models for unified text representations; sub-group-specific bias mitigation tested on MIMIC-Eye1 and MIMIC-IV ED datasets",
        "clinical_setting": "Emergency department (MIMIC-Eye1, MIMIC-IV ED)",
        "key_findings": "Demonstrated that sub-group-specific bias mitigation is robust across different datasets, subgroups, and embeddings; showed intersectional demographic groups face compounded biases; provided effective strategy for addressing intersectional biases in multimodal clinical settings."
    },
    # -------------------------------------------------------------------------
    # Paper 26
    # -------------------------------------------------------------------------
    {
        "title": "Addressing cognitive bias in medical language models",
        "authors": "Samuel Schmidgall, Carl Harris, Ime Essien, Daniel Olshvang, Tawsifur Rahman, Ji Woong Kim, Rojin Ziaei, Jason Eshraghian, Peter Abadir, Rama Chellappa",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2402.08113",
        "doi": None,
        "study_type": "Empirical study / Benchmark development",
        "ai_ml_method": "Large Language Models (GPT-4, Mixtral-8x70B, Llama 2 70B-chat, PMC Llama 13B, and others)",
        "health_domain": "Medical licensing / Clinical reasoning",
        "bias_axes": ["Cognitive biases (anchoring, availability, confirmation, etc.) in AI systems"],
        "lifecycle_stage": "Model evaluation",
        "assessment_or_mitigation": "Assessment",
        "approach_method": "Developed BiasMedQA benchmark: 1,273 USMLE questions modified to replicate common clinically-relevant cognitive biases; evaluated six LLMs to assess susceptibility to cognitive bias patterns",
        "clinical_setting": "Medical licensing examination / Clinical decision-making",
        "key_findings": "GPT-4 showed resilience to cognitive bias, while Llama 2 70B-chat and PMC Llama 13B were disproportionately affected; demonstrated that LLMs can exhibit cognitive bias patterns analogous to human clinicians; concluded bias mitigation is necessary for developing dependable medical AI systems."
    },
    # -------------------------------------------------------------------------
    # Paper 27
    # -------------------------------------------------------------------------
    {
        "title": "Bias Evaluation and Mitigation in Retrieval-Augmented Medical Question-Answering Systems",
        "authors": "Yuelyu Ji, Hang Zhang, Yanshan Wang",
        "year": 2025,
        "journal": "arXiv preprint",
        "arxiv_id": "2503.15454",
        "doi": None,
        "study_type": "Empirical study",
        "ai_ml_method": "Retrieval-Augmented Generation (RAG) with LLMs; Chain of Thought, Counterfactual filtering, Adversarial prompt refinement, Majority Vote aggregation",
        "health_domain": "Medical question-answering / Clinical decision support",
        "bias_axes": ["Race", "Gender", "Socioeconomic status"],
        "lifecycle_stage": "Model evaluation and post-processing",
        "assessment_or_mitigation": "Both",
        "approach_method": "Systematic examination of demographic biases in RAG-based medical QA across MedQA, MedMCQA, MMLU, EquityMedQA benchmarks; tested mitigation via Chain of Thought reasoning, Counterfactual filtering, Adversarial prompt refinement, and Majority Vote aggregation",
        "clinical_setting": "Medical QA systems for clinical decision support",
        "key_findings": "Found significant demographic disparities in RAG-based medical QA systems; Majority Vote aggregation delivered notable improvements in both accuracy and fairness; emphasized that equitable medical QA requires deliberate fairness-aware retrieval and strategic prompt design."
    },
    # -------------------------------------------------------------------------
    # Paper 28
    # -------------------------------------------------------------------------
    {
        "title": "Detecting Bias and Enhancing Diagnostic Accuracy in Large Language Models for Healthcare",
        "authors": "Pardis Sadat Zahraei, Zahra Shakeri",
        "year": 2024,
        "journal": "arXiv preprint",
        "arxiv_id": "2410.06566",
        "doi": None,
        "study_type": "Empirical study / Dataset and model development",
        "ai_ml_method": "Large Language Models (fine-tuned ChatDoctor -> EthiClinician; benchmarked against GPT-4)",
        "health_domain": "General healthcare diagnostics (700 diseases)",
        "bias_axes": ["Multiple (healthcare biases across demographics)"],
        "lifecycle_stage": "Model development (fine-tuning) and evaluation",
        "assessment_or_mitigation": "Both",
        "approach_method": "Developed BiasMD dataset (6,007 QA pairs) for bias evaluation/reduction and DiseaseMatcher dataset (32,000 clinical pairs, 700 diseases) for diagnostic accuracy; fine-tuned EthiClinician model from ChatDoctor",
        "clinical_setting": "General clinical diagnostics and healthcare QA",
        "key_findings": "EthiClinician outperforms GPT-4 in both ethical reasoning and clinical judgment; BiasMD dataset exposes hidden biases in existing healthcare LLMs; established standards for safer, more reliable patient outcomes through bias-aware fine-tuning."
    },
]

# -------------------------------------------------------------------------
# EXCLUDED PAPERS
# -------------------------------------------------------------------------
EXCLUDED_PAPERS = [
    {
        "arxiv_id": "2407.12680",
        "title": "Reducing Biases towards Minoritized Populations in Medical Curricular Content via Artificial Intelligence for Fairer Health Outcomes",
        "reason": "Uses AI/ML as a tool to detect bias in medical curriculum text (educational content), not about assessing or mitigating bias IN health AI/ML algorithms themselves. The subject of bias is educational materials, not AI systems."
    },
    {
        "arxiv_id": "2504.07516",
        "title": "Enhancements for Developing a Comprehensive AI Fairness Assessment Standard",
        "reason": "Proposes general AI fairness assessment standards across multiple industries (not specifically health-related); focuses on expanding the TEC Standard for tabular data, images, text, and generative AI broadly without specific health/clinical methodology."
    },
]

# -------------------------------------------------------------------------
# SUMMARY STATISTICS
# -------------------------------------------------------------------------
SCREENING_SUMMARY = {
    "total_screened": 28,
    "total_included": 26,
    "total_excluded": 2,
    "study_types": {
        "Systematic review": 4,
        "Narrative review / Survey": 5,
        "Empirical study": 8,
        "Framework proposal": 4,
        "Methodological development": 2,
        "Perspective / Commentary": 3,
    },
    "year_distribution": {
        2021: 2,
        2022: 1,
        2023: 6,
        2024: 11,
        2025: 6,
    },
    "common_bias_axes": [
        "Race/Ethnicity",
        "Gender/Sex",
        "Age",
        "Socioeconomic status",
        "Intersectional",
    ],
    "common_health_domains": [
        "General healthcare/clinical",
        "Emergency medicine",
        "Medical imaging",
        "Public health",
        "Clinical decision support",
        "Cardiology",
        "COVID-19",
    ],
}

if __name__ == "__main__":
    print(f"Total papers screened: {SCREENING_SUMMARY['total_screened']}")
    print(f"Included: {SCREENING_SUMMARY['total_included']}")
    print(f"Excluded: {SCREENING_SUMMARY['total_excluded']}")
    print(f"\nIncluded papers:")
    for i, paper in enumerate(ARXIV_RESULTS, 1):
        print(f"  {i}. [{paper['arxiv_id']}] {paper['title']} ({paper['year']})")
    print(f"\nExcluded papers:")
    for paper in EXCLUDED_PAPERS:
        print(f"  - [{paper['arxiv_id']}] {paper['title']}")
        print(f"    Reason: {paper['reason']}")
