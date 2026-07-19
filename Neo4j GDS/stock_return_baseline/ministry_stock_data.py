# -*- coding: utf-8 -*-
"""
Ministry <-> SET50 stock mapping and correlation weights.

Copied from app.py (REAL_MINISTRY_INFO / REAL_MINISTRY_STOCK_EDGES, app.py:1833-1897) rather than
imported, to keep this research pipeline independent of the Streamlit app module.

Source: Bureau of the Budget (bb.go.th / govspending.data.go.th), SET official SET50 constituent
lists, yfinance monthly closes, FY2559-2569 (2016-2026) ministry budget YoY% vs stock annual return.
"""

REAL_MINISTRY_INFO = {
    'กระทรวงการคลัง': {'id': 'finance', 'label_en': 'Finance', 'simple_r': 0.292, 'partial_r': 0.529},
    'กระทรวงพลังงาน': {'id': 'energy', 'label_en': 'Energy', 'simple_r': 0.03, 'partial_r': 0.28},
    'กระทรวงคมนาคม': {'id': 'transport', 'label_en': 'Transport', 'simple_r': 0.239, 'partial_r': 0.231},
    'กระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม': {'id': 'digital_economy', 'label_en': 'Digital Economy', 'simple_r': 0.418, 'partial_r': 0.47},
    'กระทรวงสาธารณสุข': {'id': 'public_health', 'label_en': 'Public Health', 'simple_r': 0.175, 'partial_r': 0.115},
    'กระทรวงพาณิชย์': {'id': 'commerce', 'label_en': 'Commerce', 'simple_r': -0.602, 'partial_r': -0.774},
    'กระทรวงเกษตรและสหกรณ์': {'id': 'agriculture', 'label_en': 'Agriculture', 'simple_r': -0.477, 'partial_r': -0.346},
    'กระทรวงมหาดไทย': {'id': 'interior', 'label_en': 'Interior', 'simple_r': -0.196, 'partial_r': 0.032},
    'กระทรวงอุตสาหกรรม': {'id': 'industry', 'label_en': 'Industry', 'simple_r': -0.007, 'partial_r': 0.056},
    'กระทรวงการท่องเที่ยวและกีฬา': {'id': 'tourism_and_sports', 'label_en': 'Tourism & Sports', 'simple_r': -0.061, 'partial_r': 0.097},
}

REAL_MINISTRY_STOCK_EDGES = [
    ('กระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม', 'ADVANC', 0.093),
    ('กระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม', 'TRUE', 0.55),
    ('กระทรวงคมนาคม', 'AOT', 0.415),
    ('กระทรวงคมนาคม', 'BEM', 0.084),
    ('กระทรวงคมนาคม', 'THAI', 0.004),
    ('กระทรวงมหาดไทย', 'AWC', 0.1),
    ('กระทรวงมหาดไทย', 'CPN', -0.228),
    ('กระทรวงมหาดไทย', 'LH', -0.323),
    ('กระทรวงมหาดไทย', 'SCC', -0.582),
    ('กระทรวงมหาดไทย', 'SCGP', -0.37),
    ('กระทรวงมหาดไทย', 'WHA', 0.403),
    ('กระทรวงพลังงาน', 'BANPU', 0.204),
    ('กระทรวงพลังงาน', 'BCP', 0.195),
    ('กระทรวงพลังงาน', 'EGCO', 0.497),
    ('กระทรวงพลังงาน', 'GPSC', -0.1),
    ('กระทรวงพลังงาน', 'GULF', 0.0),
    ('กระทรวงพลังงาน', 'IVL', -0.268),
    ('กระทรวงพลังงาน', 'OR', 0.146),
    ('กระทรวงพลังงาน', 'PTT', -0.597),
    ('กระทรวงพลังงาน', 'PTTEP', 0.789),
    ('กระทรวงพลังงาน', 'PTTGC', -0.458),
    ('กระทรวงพลังงาน', 'RATCH', 0.315),
    ('กระทรวงพลังงาน', 'TOP', -0.054),
    ('กระทรวงการคลัง', 'BBL', 0.38),
    ('กระทรวงการคลัง', 'KBANK', 0.581),
    ('กระทรวงการคลัง', 'KKP', 0.345),
    ('กระทรวงการคลัง', 'KTB', 0.356),
    ('กระทรวงการคลัง', 'KTC', -0.501),
    ('กระทรวงการคลัง', 'MTC', -0.127),
    ('กระทรวงการคลัง', 'SCB', 0.629),
    ('กระทรวงการคลัง', 'TCAP', 0.19),
    ('กระทรวงการคลัง', 'TIDLOR', 0.0),
    ('กระทรวงการคลัง', 'TISCO', 0.124),
    ('กระทรวงการคลัง', 'TLI', 0.441),
    ('กระทรวงการคลัง', 'TTB', 0.112),
    ('กระทรวงสาธารณสุข', 'BDMS', 0.086),
    ('กระทรวงสาธารณสุข', 'BH', 0.212),
    ('กระทรวงพาณิชย์', 'BJC', -0.402),
    ('กระทรวงพาณิชย์', 'COM7', -0.406),
    ('กระทรวงพาณิชย์', 'CPALL', -0.4),
    ('กระทรวงพาณิชย์', 'CRC', -0.631),
    ('กระทรวงพาณิชย์', 'HMPRO', -0.583),
    ('กระทรวงพาณิชย์', 'MRDIYT', 0.0),
    ('กระทรวงอุตสาหกรรม', 'CCET', -0.065),
    ('กระทรวงอุตสาหกรรม', 'DELTA', 0.017),
    ('กระทรวงเกษตรและสหกรณ์', 'CPF', -0.313),
    ('กระทรวงเกษตรและสหกรณ์', 'OSP', -0.493),
    ('กระทรวงเกษตรและสหกรณ์', 'TFG', -0.343),
    ('กระทรวงเกษตรและสหกรณ์', 'TU', 0.022),
    ('กระทรวงการท่องเที่ยวและกีฬา', 'MINT', -0.061),
]
