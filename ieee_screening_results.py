"""
IEEE Xplore Screening Results for Systematic Review:
"Approaches for Assessing and Mitigating Algorithmic Bias in Health AI"

Screening Date: 2026-02-16
Total papers screened: 31
Included: 19
Excluded: 12

EXCLUSION LOG:
  Paper 4  - "Legal and Ethical Challenges of AI in Healthcare" -> EXCLUDE: General legal/ethical discussion, not focused on bias assessment/mitigation methods
  Paper 5  - "Ethical Challenges and Solutions in AI-Powered Digital Health" -> EXCLUDE: Broad ethical overview, not specific bias assessment/mitigation methodology
  Paper 6  - "Predictive Analytics in Healthcare using Machine Learning" -> EXCLUDE: General ML in healthcare; bias only briefly mentioned as a challenge
  Paper 8  - "AI Algorithmic Bias: Understanding its Causes, Ethical and Social Implications" -> EXCLUDE: General AI bias paper, not health/healthcare specific
  Paper 9  - "Ethical Dimensions of AI in Personalized Medicine" -> EXCLUDE: Broad ethics survey of personalized medicine, not specific bias assessment/mitigation methods
  Paper 15 - "Guest Editorial Explainable AI: Towards Fairness, Accountability in Healthcare" -> EXCLUDE: Guest editorial, not a research article with bias assessment/mitigation methods
  Paper 20 - "Is data quality enough for a clinical decision?" -> EXCLUDE: Focuses on data quality assurance, not algorithmic bias/fairness in AI
  Paper 21 - "AI Ethics and Bias in ML Models in AR/VR" -> EXCLUDE: Not health-specific; covers AR/VR across education, healthcare, finance broadly
  Paper 22 - "Hounsfield Unit Ranges as Inductive Bias for CT Segmentation" -> EXCLUDE: Inductive bias in model architecture (not algorithmic fairness/demographic bias)
  Paper 23 - "Machine Learning in Healthcare: A Review" -> EXCLUDE: General ML review with no bias/fairness focus
  Paper 25 - "Clinical big data and deep learning" -> EXCLUDE: General deep learning in healthcare overview, no bias/fairness focus
  Paper 27 - "Maximizing sensitivity in medical diagnosis using biased minimax probability Machine" -> EXCLUDE: Statistical/class bias for sensitivity optimization, not demographic algorithmic fairness
  Paper 28 - "ML and Deep Learning Approaches for Healthcare Predictive Analytics" -> EXCLUDE: General ML/DL predictive analytics review, no bias/fairness focus
  Paper 29 - "Randomized Explainable ML Models for Efficient Medical Diagnosis" -> EXCLUDE: Explainability focus, not bias/fairness assessment or mitigation
"""

IEEE_RESULTS = [
    {
        # Paper 1
        "title": "Mitigating Racial Algorithmic Bias in Healthcare Artificial Intelligent Systems: A Fairness-Aware Machine Learning Approach",
        "authors": "Nyawambi et al.",
        "year": 2024,
        "journal_conference": "2024 5th International Conference on Smart Sensors and Application (ICSSA)",
        "doi": "10.1109/ICSSA62312.2024.10788666",
        "study_type": "Empirical study",
        "ai_ml_method": "Logistic Regression, Random Forest with Reweighing, Adversarial Debiasing, Exponentiated Gradient; LIME for interpretability",
        "health_domain": "Healthcare access",
        "bias_axes": "Race",
        "lifecycle_stage": "Pre-processing (Reweighing), In-processing (Adversarial Debiasing, Exponentiated Gradient)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Reweighing, Adversarial Debiasing, Exponentiated Gradient bias mitigation integrated with standard ML classifiers; fairness metrics evaluation",
        "clinical_setting": "Healthcare access prediction",
        "key_findings": "Random Forest with Reweighing achieved 88.01% accuracy, 89% precision, 97% recall, 93% F1 while maintaining equity across racial groups. Demonstrated that integrating bias-mitigating algorithms with standard classifiers can reduce racial bias without major performance loss."
    },
    {
        # Paper 2
        "title": "Algorithmic Bias in Clinical Populations—Evaluating and Improving Facial Analysis Technology in Older Adults With Dementia",
        "authors": "Taati et al.",
        "year": 2019,
        "journal_conference": "IEEE Access",
        "doi": "10.1109/ACCESS.2019.2900022",
        "study_type": "Empirical study",
        "ai_ml_method": "Facial expression analysis (landmark placement algorithms), retraining with domain-specific data",
        "health_domain": "Dementia / geriatric care",
        "bias_axes": "Age, cognitive status (dementia vs. cognitively healthy)",
        "lifecycle_stage": "Training data (retraining with representative data)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Evaluation of facial landmark placement accuracy across clinical populations; retraining models with images of older adults to mitigate bias",
        "clinical_setting": "Facial expression analysis for pain assessment in older adults with dementia",
        "key_findings": "Facial landmark placement is significantly less accurate for individuals with dementia compared to cognitively healthy older adults. Retraining with images of older adults' faces improves performance, but disparities between healthy and impaired groups persist."
    },
    {
        # Paper 3
        "title": "Framework for Algorithmic Bias Quantification and its Application to Automated Sleep Scoring",
        "authors": "Bechny et al.",
        "year": 2024,
        "journal_conference": "2024 11th IEEE Swiss Conference on Data Science (SDS)",
        "doi": "10.1109/SDS60720.2024.00045",
        "study_type": "Methodological / framework development",
        "ai_ml_method": "GAMLSS (Generalized Additive Models for Location, Scale, and Shape) for bias quantification of automated sleep scoring algorithms",
        "health_domain": "Sleep medicine",
        "bias_axes": "Systematic algorithmic error across external factors (bias-generating factors identified via GAMLSS)",
        "lifecycle_stage": "Post-processing (evaluation/quantification of bias in predictions)",
        "assessment_or_mitigation": "Assessment",
        "approach_method": "GAMLSS-based framework for in-depth quantification of systematic algorithmic error (bias), estimating bias distribution, identifying bias-generating factors, and extrapolating prediction validity quantities",
        "clinical_setting": "Automated sleep scoring / polysomnography",
        "key_findings": "Proposed a flexible statistical framework using GAMLSS that goes beyond traditional Bland-Altman analysis to quantify algorithmic bias without simplifying assumptions. Enables identification of factors generating bias in automated sleep scoring."
    },
    {
        # Paper 7
        "title": "Integrated Framework for Equitable Healthcare AI: Bias Mitigation, Community Participation, and Regulatory Governance",
        "authors": "Chandra et al.",
        "year": 2025,
        "journal_conference": "2025 IEEE 14th International Conference on Communication Systems and Network Technologies (CSNT)",
        "doi": "10.1109/CSNT64827.2025.10968102",
        "study_type": "Framework development with empirical validation",
        "ai_ml_method": "Adversarial debiasing, reinforcement learning-driven threshold optimization, synthetic data augmentation",
        "health_domain": "General healthcare (ICU / critical care via MIMIC-III)",
        "bias_axes": "Demographic parity (race, gender implied)",
        "lifecycle_stage": "Pre-processing (synthetic data augmentation), In-processing (adversarial debiasing, RL threshold optimization)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Combined adversarial debiasing, RL-driven threshold optimization, synthetic data augmentation, and community engagement; aligned with EU AI Act governance framework",
        "clinical_setting": "Clinical prediction using MIMIC-III dataset",
        "key_findings": "Achieved near-perfect AUC (0.994) alongside significant reductions in demographic parity disparity on MIMIC-III. Demonstrated that combining technical debiasing with participatory design and regulatory alignment can yield equitable healthcare AI."
    },
    {
        # Paper 10
        "title": "Exploring Bias and Prediction Metrics to Characterise the Fairness of Machine Learning for Equity-Centered Public Health Decision-Making: A Narrative Review",
        "authors": "Raza et al.",
        "year": 2024,
        "journal_conference": "IEEE Access",
        "doi": "10.1109/ACCESS.2024.3509353",
        "study_type": "Narrative review",
        "ai_ml_method": "Various ML methods (reviewed across 72 included articles)",
        "health_domain": "Public and population health",
        "bias_axes": "Multiple (equity-focused: race, gender, socioeconomic status, and others as reviewed)",
        "lifecycle_stage": "All stages (comprehensive review of bias types and metrics)",
        "assessment_or_mitigation": "Assessment",
        "approach_method": "Narrative review of bias types and quantitative metrics for assessing ML bias in public health; evaluation framework formalization from equity perspective",
        "clinical_setting": "Public health surveillance and decision-making",
        "key_findings": "Identified commonly described types of bias and quantitative metrics to assess them from an equity perspective across 72 articles. Formalized an evaluation framework for ML fairness in public health applications."
    },
    {
        # Paper 11
        "title": "A Survey of Bias and Fairness in Healthcare AI",
        "authors": "Mienye et al.",
        "year": 2024,
        "journal_conference": "2024 IEEE 12th International Conference on Healthcare Informatics (ICHI)",
        "doi": "10.1109/ICHI61247.2024.00103",
        "study_type": "Survey / review",
        "ai_ml_method": "Various AI/ML methods (surveyed)",
        "health_domain": "General healthcare",
        "bias_axes": "Multiple (general review across patient populations)",
        "lifecycle_stage": "All stages (comprehensive survey)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Survey of bias sources, fairness definitions, and ethical considerations in healthcare AI systems",
        "clinical_setting": "General healthcare AI applications",
        "key_findings": "Provided a foundation for future research and policy development by reviewing key issues of bias, fairness, and ethical considerations in healthcare AI, identifying concerns about discrimination and unfair practices."
    },
    {
        # Paper 12
        "title": "Fairness Artificial Intelligence in Clinical Decision Support: Mitigating Effect of Health Disparity",
        "authors": "Wu et al.",
        "year": 2025,
        "journal_conference": "IEEE Journal of Biomedical and Health Informatics",
        "doi": "10.1109/JBHI.2024.3513398",
        "study_type": "Empirical study / method development",
        "ai_ml_method": "CFReg (causal-model-free counterfactual fairness algorithm) for supervised learning",
        "health_domain": "Clinical decision support / care management",
        "bias_axes": "Race (and other sensitive attributes)",
        "lifecycle_stage": "In-processing (counterfactual fairness regularization)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Causal inference-based counterfactual fairness (CFReg) with novel clinical fairness evaluation metric; addresses whether CDS would reach different conclusions if patient had different sensitive attribute",
        "clinical_setting": "Care management (48,784 patient records)",
        "key_findings": "CFReg outperforms baseline approaches in both fairness and accuracy on 48,784 patient records and four benchmark datasets. Demonstrated that causal counterfactual approach achieves better balance between model fairness and classification performance than correlation-based methods."
    },
    {
        # Paper 13
        "title": "Fairness Metrics in AI Healthcare Applications: A Review",
        "authors": "Mienye et al.",
        "year": 2024,
        "journal_conference": "2024 IEEE International Conference on Information Reuse and Integration for Data Science (IRI)",
        "doi": "10.1109/IRI62200.2024.00065",
        "study_type": "Survey / review",
        "ai_ml_method": "Various AI methods (reviewed across healthcare applications)",
        "health_domain": "General healthcare",
        "bias_axes": "Multiple (diverse patient populations)",
        "lifecycle_stage": "Post-processing / evaluation (fairness metrics focus)",
        "assessment_or_mitigation": "Assessment",
        "approach_method": "Survey of fairness metrics including mathematical representations, suitable use cases, and limitations for healthcare AI applications",
        "clinical_setting": "Various healthcare AI applications",
        "key_findings": "Provided a concise catalogue of fairness metrics with mathematical representations, suitable use cases, and limitations for healthcare AI. Highlighted the significance of implementing fairness metrics to ensure equitable outcomes across diverse patient populations."
    },
    {
        # Paper 14
        "title": "Fairness in Healthcare AI",
        "authors": "Ahmad et al.",
        "year": 2021,
        "journal_conference": "2021 IEEE 9th International Conference on Healthcare Informatics (ICHI)",
        "doi": "10.1109/ICHI52183.2021.00104",
        "study_type": "Tutorial / review",
        "ai_ml_method": "Various ML methods (discussed conceptually)",
        "health_domain": "General healthcare",
        "bias_axes": "Multiple (general fairness concepts in healthcare)",
        "lifecycle_stage": "All stages (conceptual overview)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Tutorial examining how ML fairness concepts map to fairness notions in medical contexts, with real-world examples",
        "clinical_setting": "General healthcare AI applications",
        "key_findings": "Examined challenges, requirements, and opportunities for fairness in healthcare AI. Demonstrated how standard ML fairness concepts correspond to fairness notions in medical contexts through real-world examples."
    },
    {
        # Paper 16
        "title": "Fairness in Healthcare: Assessing Data Bias and Algorithmic Fairness",
        "authors": "Dehghani et al.",
        "year": 2024,
        "journal_conference": "2024 20th International Symposium on Medical Information Processing and Analysis (SIPAIM)",
        "doi": "10.1109/SIPAIM62974.2024.10783630",
        "study_type": "Empirical study",
        "ai_ml_method": "ML predictive models with imbalance handling techniques",
        "health_domain": "General healthcare",
        "bias_axes": "Demographic groups (assessed via publicly available healthcare datasets)",
        "lifecycle_stage": "Pre-processing (data bias identification, imbalance handling), Post-processing (fairness evaluation)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Investigation of publicly available healthcare datasets for bias; evaluation of imbalance handling techniques on model performance and fairness",
        "clinical_setting": "Healthcare predictive modeling (public datasets)",
        "key_findings": "Bias in healthcare datasets exerts a negative impact on model fairness. Various imbalance handling techniques have differential impacts on both performance and fairness. Underscored the importance of early bias detection to mitigate risks in healthcare applications."
    },
    {
        # Paper 17
        "title": "Gradient-Based Reconciliation of Fairness and Performance in Healthcare AI",
        "authors": "Wang et al.",
        "year": 2025,
        "journal_conference": "2025 IEEE 13th International Conference on Healthcare Informatics (ICHI)",
        "doi": "10.1109/ICHI64645.2025.00081",
        "study_type": "Method development with empirical validation",
        "ai_ml_method": "Gradient projection for multi-attribute fairness optimization",
        "health_domain": "General healthcare",
        "bias_axes": "Multiple demographic attributes (multi-attribute)",
        "lifecycle_stage": "In-processing (gradient-based optimization during training)",
        "assessment_or_mitigation": "Mitigation",
        "approach_method": "Gradient projection method aligning conflicting optimization objectives (fairness vs. performance) by projecting each gradient onto the normal plane of the other; enhances interpretability of fairness-accuracy trade-offs",
        "clinical_setting": "Healthcare predictive modeling",
        "key_findings": "Novel gradient projection approach concurrently optimizes fairness across multiple demographic attributes and predictive performance. Method enhances interpretability by elucidating adjustments made during optimization, providing insights into fairness-accuracy trade-offs."
    },
    {
        # Paper 18
        "title": "Enhancing Multi-Attribute Fairness in Healthcare Predictive Modeling",
        "authors": "Wang et al.",
        "year": 2025,
        "journal_conference": "2025 IEEE 13th International Conference on Healthcare Informatics (ICHI)",
        "doi": "10.1109/ICHI64645.2025.00015",
        "study_type": "Method development with empirical validation",
        "ai_ml_method": "Two-phase optimization (accuracy maximization then fairness fine-tuning); sequential and simultaneous optimization strategies",
        "health_domain": "General healthcare",
        "bias_axes": "Multiple demographic attributes simultaneously",
        "lifecycle_stage": "In-processing (multi-attribute fairness optimization during training)",
        "assessment_or_mitigation": "Mitigation",
        "approach_method": "Two-phase methodology: first maximize predictive accuracy, then fine-tune for multi-attribute fairness using sequential and simultaneous optimization strategies",
        "clinical_setting": "Healthcare predictive modeling",
        "key_findings": "Focusing on single-attribute fairness can inadvertently worsen disparities in other attributes. Simultaneous multi-attribute approach achieves substantial reductions in Equalized Odds Disparity while maintaining strong accuracy and more equitable outcomes across all demographic groups."
    },
    {
        # Paper 19
        "title": "Post-Processing Fairness Evaluation of Federated Models: An Unsupervised Approach in Healthcare",
        "authors": "Siniosoglou et al.",
        "year": 2023,
        "journal_conference": "IEEE/ACM Transactions on Computational Biology and Bioinformatics",
        "doi": "10.1109/TCBB.2023.3269767",
        "study_type": "Method development with empirical validation",
        "ai_ml_method": "Federated Learning with unsupervised fairness evaluation via micro-manifold analysis of latent knowledge",
        "health_domain": "General healthcare (medical cyberphysical systems)",
        "bias_axes": "Model-level fairness (data and model agnostic)",
        "lifecycle_stage": "Post-processing (fairness evaluation of trained federated models)",
        "assessment_or_mitigation": "Assessment",
        "approach_method": "Unsupervised, model-agnostic and data-agnostic methodology for ranking federated models by fairness through examination of micro-manifolds in neural models' latent knowledge",
        "clinical_setting": "Healthcare federated learning systems",
        "key_findings": "Proposed an unsupervised post-processing method for evaluating fairness in federated healthcare models. Achieved approximately 8.75% improvement in federated model accuracy compared to similar approaches across various deep learning architectures."
    },
    {
        # Paper 24
        "title": "Addressing Racial Bias in Cardiovascular Disease Risk Prediction with Fair Data Augmentation",
        "authors": "Cui et al.",
        "year": 2023,
        "journal_conference": "2023 International Conference on Computational Intelligence, Networks and Security (ICCINS)",
        "doi": "10.1109/ICCINS58907.2023.10450015",
        "study_type": "Empirical study",
        "ai_ml_method": "ML classifiers with Fair Mixup data augmentation",
        "health_domain": "Cardiovascular disease",
        "bias_axes": "Race",
        "lifecycle_stage": "Pre-processing (fair data augmentation)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Fair Mixup data augmentation technique applied to CDC survey data to improve fairness metrics between racial groups in CVD risk prediction",
        "clinical_setting": "Cardiovascular disease risk prediction (CDC survey data)",
        "key_findings": "Fair Mixup data augmentation improves fairness metrics between racial groups while enhancing predictive accuracy across populations in cardiovascular disease risk prediction."
    },
    {
        # Paper 26
        "title": "Bias Reducing Multitask Learning on Mental Health Prediction",
        "authors": "Zanna et al.",
        "year": 2022,
        "journal_conference": "2022 10th International Conference on Affective Computing and Intelligent Interaction (ACII)",
        "doi": "10.1109/ACII55700.2022.9953850",
        "study_type": "Empirical study",
        "ai_ml_method": "Multi-task learning with epistemic uncertainty for bias mitigation; ECG-based physiological signal analysis",
        "health_domain": "Mental health (anxiety prediction)",
        "bias_axes": "Age, income, ethnicity, birthplace",
        "lifecycle_stage": "In-processing (multi-task learning approach for bias reduction)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Multi-task learning approach based on epistemic uncertainty principles for bias mitigation in anxiety prediction from ECG data; compared against reweighting techniques",
        "clinical_setting": "Mental health anxiety detection from physiological signals (ECG)",
        "key_findings": "Baseline model exhibited bias across age, income, ethnicity, and birthplace. Proposed multi-task learning bias mitigation method outperformed reweighting techniques. Feature importance analysis revealed connections between heart rate variability and demographic characteristics."
    },
    {
        # Paper 30
        "title": "A Taxonomy of Machine Learning Fairness Tool Specifications, Features and Workflows",
        "authors": "Mim et al.",
        "year": 2023,
        "journal_conference": "2023 IEEE Symposium on Visual Languages and Human-Centric Computing (VL/HCC)",
        "doi": "10.1109/VL-HCC57772.2023.00036",
        "study_type": "Taxonomy / tool evaluation",
        "ai_ml_method": "14 ML fairness tools evaluated (various)",
        "health_domain": "Healthcare (among other domains including criminal justice)",
        "bias_axes": "Multiple (tool-dependent)",
        "lifecycle_stage": "All stages (tools span pre-processing, in-processing, post-processing)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Evaluation and categorization of 14 existing fairness tools based on features and usage workflows to develop a practical taxonomy",
        "clinical_setting": "General (healthcare as key application domain)",
        "key_findings": "Developed a taxonomy of 14 ML fairness tools categorized by features and workflows. Found availability of tools ranging from low-code to highly customizable options. Provides guidance for practitioners to select appropriate fairness tools."
    },
    {
        # Paper 31
        "title": "Ethical AI with Balancing Bias Mitigation and Fairness in Machine Learning Models",
        "authors": "Nathim et al.",
        "year": 2024,
        "journal_conference": "2024 36th Conference of Open Innovations Association (FRUCT)",
        "doi": "10.23919/FRUCT64283.2024.10749873",
        "study_type": "Systematic review with empirical experiments",
        "ai_ml_method": "Various ML models across 25 benchmark datasets; bias mitigation techniques during model training",
        "health_domain": "General (includes healthcare applications among benchmarks)",
        "bias_axes": "Multiple (9 fairness metrics evaluated)",
        "lifecycle_stage": "In-processing (bias mitigation during model training)",
        "assessment_or_mitigation": "Both",
        "approach_method": "Systematic review of 150 articles (2018-2023) plus experiments on 25 benchmark datasets evaluating bias mitigation approaches and fairness-performance trade-offs",
        "clinical_setting": "General ML applications (includes healthcare benchmarks)",
        "key_findings": "Achieved 23% reduction in bias and average 17% improvement across 9 fairness metrics during model training, but accuracy decreased up to 9%. Highlighted trade-offs between fairness and performance, emphasizing need for adaptive frameworks."
    },
]
