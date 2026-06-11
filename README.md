# SET50 Shareholder Network Analysis

Streamlit app สำหรับดึงรายชื่อผู้ถือหุ้นรายใหญ่ของบริษัทใน SET50 จากเว็บไซต์ SET แล้วสร้าง Social Network Analysis เพื่อดูว่าใครถือหุ้นบริษัทไหนบ้าง และบริษัทใดเชื่อมกันผ่านผู้ถือหุ้นร่วม

## สิ่งที่แอปทำ

- ดึงรายชื่อหุ้นใน SET50 ล่าสุดจากหน้า `SET50 overview`
- ดึงรายชื่อผู้ถือหุ้นรายใหญ่ของแต่ละบริษัทจากหน้า `major shareholders`
- สร้างกราฟแบบ bipartite ระหว่าง `shareholder -> company`
- คำนวณตัวชี้วัดพื้นฐาน เช่น degree, company overlap, connected components
- แสดงผลผ่าน Streamlit พร้อมตารางและ network graph

## ติดตั้ง

```bash
pip install -r requirements.txt
playwright install chromium
```

## รัน

```bash
streamlit run app.py
```

## หมายเหตุสำคัญ

- ข้อมูลอ้างอิงจากหน้าเว็บ SET ที่เปิดเผยต่อสาธารณะ
- หน้า SET โหลดข้อมูลด้วย JavaScript ดังนั้นแอปใช้ Playwright เพื่อ render หน้าเว็บก่อนดึงข้อมูล
- ผู้ถือหุ้นบางรายเป็น nominee/custodian เช่น `THAI NVDR`, `STATE STREET`, `CITIBANK NOMINEES` จึงมีตัวเลือกให้ซ่อนเพื่อให้เห็นความสัมพันธ์เชิงโครงสร้างชัดขึ้น
- ชื่อผู้ถือหุ้นถูก normalize ระดับหนึ่งเพื่อรวมชื่อที่สะกดคล้ายกัน แต่ยังไม่ใช่ entity resolution แบบสมบูรณ์
