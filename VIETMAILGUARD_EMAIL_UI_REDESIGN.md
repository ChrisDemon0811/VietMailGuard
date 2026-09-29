# VietMailGuard — Email UI Redesign Brief & Codex Prompt

## 1. Mục tiêu

Thiết kế lại **giao diện Email Client** của VietMailGuard theo hướng:

- đơn giản;
- sạch;
- dễ đọc;
- giống một email client cơ bản hơn là một cybersecurity dashboard;
- ít màu;
- ít hiệu ứng;
- ưu tiên tốc độ quét danh sách email và đọc nội dung;
- vẫn giữ các cảnh báo spam/phishing khi thật sự cần thiết.

Không cần giao diện “wow”, glassmorphism, gradient, animation lớn, card dày đặc hoặc dashboard nhiều metric.

---

## 2. Nguồn định hướng UI/UX

### UI UX Pro Max

Repo tham khảo:

`https://github.com/nextlevelbuilder/ui-ux-pro-max-skill`

Các nguyên tắc phù hợp nhất cho **Email Client** trong repo:

- Style chính: **Flat Design + Minimalism & Swiss Style**.
- Màu sắc: nền trắng/sáng, một màu brand chính, đỏ cho priority/cảnh báo, amber cho trạng thái cần chú ý.
- Typography: chuyên nghiệp, sạch, hierarchy rõ.
- Interaction: hover nhẹ, transition nhanh khoảng 150 ms.
- Tránh:
  - excessive decoration;
  - complex shadows;
  - 3D effects;
  - quá nhiều màu;
  - giao diện mang cảm giác marketing/dashboard.
- Accessibility:
  - contrast văn bản rõ;
  - visible keyboard focus;
  - trạng thái không chỉ dựa vào màu;
  - text/badge không bị clipping;
  - responsive ở các kích thước nhỏ và lớn.

### OpenAI prompting guidance

Tài liệu tham khảo:

`https://learn.chatgpt.com/docs/prompting`

Prompt triển khai bên dưới được cấu trúc theo:

1. **Goal** — kết quả cần đạt.
2. **Context** — code và kiến trúc hiện tại.
3. **Output** — những file/kết quả cần thay đổi.
4. **Boundaries** — phần không được thay đổi.
5. **Verification** — cách kiểm tra sau khi sửa.

Đối với code task, prompt chỉ rõ behavior mong muốn, file liên quan, constraint và test/verification.

---

## 3. Hiện trạng VietMailGuard cần đơn giản hóa

Các file UI chính hiện tại:

- `app/app.py`
- `app/pages/mail_client.py`
- `app/mailbox_ui.py`
- `app/styles.py`
- `app/i18n.py`

Các test liên quan nên kiểm tra:

- `tests/test_mailbox_ui.py`
- `tests/test_streamlit_app.py`
- `tests/test_mail_service.py`
- `tests/test_mail_router.py`
- `tests/test_mail_repository.py`

### Một số điểm UI hiện tại nên thay đổi

#### Header

Hiện có kiểu header mang tính marketing:

`AI-POWERED BILINGUAL EMAIL SECURITY CLIENT`

Không cần thiết đối với một email client cơ bản.

Nên chuyển thành header đơn giản:

- `Inbox`
- số thư / số chưa đọc nhỏ bên cạnh nếu cần;
- refresh là icon/button phụ;
- không hero;
- không slogan lớn.

#### Danh sách email

Hiện sender + subject + preview được ghép thành một button lớn.

Nên đổi thành row email giống mailbox truyền thống:

```text
☆  John Doe     Project meeting tomorrow
                Can we move the meeting to 2 PM?              10:42

★  Security     Verify your account immediately        PHISHING
                Your account will be locked...                  09:15
```

Hierarchy:

1. Star.
2. Sender.
3. Subject.
4. Preview.
5. Status nhỏ nếu cần.
6. Timestamp.

Unread:

- sender/subject semibold;
- có thể có một dot nhỏ;
- không cần biến cả row thành primary button màu nổi.

#### Search / Import / Filter

Thanh công cụ nên gọn:

```text
Inbox

[ Search mail................................ ]  [Filter]  [Import .eml]
```

- Search là thành phần chính.
- Filter và Import là secondary actions.
- Không để uploader chiếm nhiều diện tích.
- Advanced filters có thể nằm trong expander/popover hiện có.

#### Sidebar

Mailbox navigation nên là nội dung chính:

```text
VietMailGuard Mail

Inbox
Starred
Spam
Quarantine
Deleted

Security View

────────────

Tools
Quick Analyzer
Model Dashboard
Dataset Dashboard
Methodology
```

- Language selector đặt gọn ở đầu hoặc cuối sidebar.
- Không cần nhiều caption/slogan.
- Counts hiển thị nhỏ bên phải hoặc cuối label.
- Inbox unread nên dễ nhìn nhưng không gây nhiễu.

#### Email detail

Cấu trúc mong muốn:

```text
← Back to Inbox

Project meeting tomorrow                         [NORMAL]

John Doe <john@example.com>
to me
Sep 25, 2026 · 10:42

[Star] [Unread] [Spam] [Report phishing] [Delete]
--------------------------------------------------

Email content...

--------------------------------------------------

Security analysis
▸ Model & risk details
▸ Security findings
▸ URLs
▸ Sender findings
▸ Folder history
```

Nguyên tắc:

- nội dung email phải là trọng tâm;
- security analysis đặt sau nội dung;
- advanced details collapsed mặc định;
- alert phishing/quarantine vẫn nổi bật;
- normal email không cần quá nhiều green badge;
- destructive actions không được visually equal với action thường.

---

## 4. Visual system đề xuất

### Color tokens

```text
Background:       #FFFFFF
Page background:  #F8FAFC
Primary:          #2563EB
Primary hover:    #1D4ED8

Text:             #0F172A
Muted text:       #64748B
Border:           #E2E8F0
Hover background: #F8FAFC
Selected row:     #EFF6FF

Normal:           #15803D
Spam / warning:   #D97706
Phishing / danger:#DC2626
```

Không dùng thêm màu nếu không cần.

### Typography

Ưu tiên system font / sans-serif mặc định.

Gợi ý hierarchy:

```text
Page title:      24–28 px, 600
Section title:   18–20 px, 600
Sender:          14–15 px, 600 nếu unread
Subject:         14–15 px, 500/600
Preview:         13–14 px, 400, muted
Metadata:        12–13 px, muted
Badge:           11–12 px, 600
```

### Shape

```text
Border radius: 6–8 px
Border:        1 px solid #E2E8F0
Shadow:        none
Gradient:      none
Blur:          none
```

Không lạm dụng pill. Chỉ badge/status nhỏ mới cần pill.

### Motion

- 120–180 ms cho hover/state transition.
- Không animation liên tục.
- Tôn trọng `prefers-reduced-motion`.
- Không scale/bounce card.

---

## 5. UX rules

1. Email list phải scan nhanh bằng mắt.
2. Sender, subject, preview và time phải có hierarchy rõ.
3. Phishing/spam phải nhận biết được ngoài màu sắc.
4. Normal không cần cảnh báo nổi bật.
5. Click target phải đủ lớn nhưng không biến từng email thành button xanh.
6. Search luôn dễ thấy.
7. Filter phụ, không cạnh tranh với search.
8. Nội dung email ưu tiên hơn ML/security metrics.
9. Các thông tin kỹ thuật để trong expander.
10. Không tự động mở URL trong email.
11. Không thay đổi routing hoặc security behavior hiện có.
12. Responsive và không clipping text.

---

# 6. Prompt hoàn chỉnh để đưa cho Codex / ChatGPT coding agent

```text
You are redesigning ONLY the email/mailbox user interface of the existing
VietMailGuard repository.

GOAL

Make VietMailGuard Mail look and behave like a simple, clean, conventional
email client.

The desired UI is intentionally restrained. I do NOT want a flashy
cybersecurity dashboard, marketing-style hero, glassmorphism, gradients,
3D effects, large cards, heavy shadows, or excessive colors.

The design direction is:

- Flat Design
- Minimalism / Swiss-style clarity
- clean white or near-white surfaces
- one primary blue accent
- red reserved for phishing/destructive warnings
- amber reserved for spam/warnings
- clear typography and spacing
- fast, subtle interactions
- no decorative visual effects

The email content and mailbox workflow are more important than showcasing
the machine-learning system.

CONTEXT

This is an existing Streamlit application. Do not rewrite it into React,
Next.js, Tailwind, or another frontend framework.

Before editing, inspect at minimum:

- app/app.py
- app/pages/mail_client.py
- app/mailbox_ui.py
- app/styles.py
- app/i18n.py

Also inspect relevant tests, especially:

- tests/test_mailbox_ui.py
- tests/test_streamlit_app.py
- tests/test_mail_service.py
- tests/test_mail_router.py
- tests/test_mail_repository.py

Understand the existing list/detail navigation and current MailService API
before changing the UI.

IMPORTANT ARCHITECTURE BOUNDARIES

Preserve the existing application architecture:

Streamlit -> MailService -> MailRepository / frozen inference -> SQLite

Do NOT change:

- ML model or inference behavior
- prediction labels
- model confidence calculation
- risk score calculation
- security rules
- email routing behavior
- SQLite schema unless absolutely necessary (it should not be necessary)
- repository/service responsibilities
- stored analysis behavior
- bilingual i18n behavior
- query-parameter/session-state list/detail navigation semantics
- folder meanings: Inbox, Spam, Quarantine, Deleted
- existing email import logic
- security behavior that prevents unsafe automatic link/image loading

This task is a UI/UX refactor, not an ML/backend refactor.

DESIGN TARGET

1. SIDEBAR

Make the sidebar look like a basic mailbox navigation.

Preferred structure:

VietMailGuard Mail

Inbox
Starred
Spam
Quarantine
Deleted

Security View

----------------

Tools
Quick Analyzer
Model Dashboard
Dataset Dashboard
Methodology

Keep language selection compact.

Remove or reduce marketing-style captions and unnecessary explanatory text
from the primary mailbox navigation.

Mailbox counts may remain, but they should be visually secondary.
Unread Inbox count should still be easy to notice.

2. MAILBOX PAGE HEADER

Remove the marketing-style hero/eyebrow such as:

"AI-POWERED BILINGUAL EMAIL SECURITY CLIENT"

The header should be compact and functional.

Example:

Inbox                                      [Refresh]

24 messages · 3 unread

Do not use a large hero component.

3. TOOLBAR

Create a compact toolbar with search as the primary control.

Desired hierarchy:

[ Search mail................................ ] [Filters] [Import .eml]

Search should receive most horizontal space.

Import must be a secondary action and should not dominate the page.
Reuse Streamlit-native components wherever practical.

Advanced filters can stay collapsed until requested.

4. EMAIL LIST

This is the most important change.

Redesign each email as a compact mailbox row rather than a large primary or
tertiary button containing all text.

Each row should visually communicate:

- star
- sender
- subject
- body preview
- optional small prediction/risk status
- timestamp

Target hierarchy:

STAR | SENDER | SUBJECT
                preview text                       STATUS | TIME

For unread messages:

- use semibold sender and subject
- optionally show a small unread dot
- do not fill the entire row with strong primary blue

For read messages:

- normal font weight
- slightly muted appearance

Use thin separators or a 1 px bottom border.

On hover, use only a subtle background such as #F8FAFC.

Selected/active state may use a very light blue such as #EFF6FF.

Do not add card shadows around every message.

Prediction badges should be compact and visually secondary.

Normal:
- neutral or subtle green indicator

Spam:
- amber indicator with text/icon meaning

Phishing:
- red indicator with text/icon meaning

Do not communicate the status through color alone.

Long sender names, subjects, URLs, and preview text must not break the layout.
Use wrapping or safe truncation with accessible full text where possible.

5. EMAIL DETAIL VIEW

Make the email itself the primary content.

Preferred structure:

← Back to Inbox

Subject                                               [status]

Sender
Recipient / timestamp / CC metadata

compact action toolbar

--------------------------------------------------

email body

--------------------------------------------------

Security analysis
collapsed technical sections

Actions may include:
- Star / Unstar
- Mark read / unread
- Move to Inbox
- Not spam / Mark spam
- Report phishing
- Quarantine
- Delete

Do not present every action as an equally prominent full-width button.

Normal actions should be compact.
Destructive or dangerous-state actions should be visually distinct but not
overly dramatic.

Keep phishing/quarantine warning banners prominent when they are relevant.
Spam can use a restrained warning.
Normal emails should not look like security incidents.

6. SECURITY ANALYSIS

Keep all existing information and behavior, but reduce its visual priority.

The email body must appear before detailed ML/security information.

Keep:
- prediction
- confidence
- risk score
- recommended action
- model explanation
- content findings
- URL findings
- sender findings
- limitations
- folder history

Prefer collapsed expanders for technical detail.

Do not make confidence/risk metrics look like dashboard KPI cards unless
there is a strong functional reason.

7. VISUAL TOKENS

Use a restrained palette close to:

Background:       #FFFFFF
Page background:  #F8FAFC
Primary:          #2563EB
Primary hover:    #1D4ED8
Text:             #0F172A
Muted text:       #64748B
Border:           #E2E8F0
Hover background: #F8FAFC
Selected row:     #EFF6FF
Normal:           #15803D
Warning / spam:   #D97706
Danger / phishing:#DC2626

Use these as guidance, not as a reason to add color everywhere.

Use:

- 6–8 px corner radius for normal controls
- 1 px neutral borders
- no gradient
- no backdrop blur
- no complex shadow
- no 3D
- no neon
- no oversized typography
- no excessive pill-shaped components

System sans-serif fonts are preferred.

8. ACCESSIBILITY

Maintain or improve accessibility.

Check:

- readable contrast (target at least 4.5:1 for normal text)
- visible keyboard focus
- meaningful labels for controls
- status meaning not encoded only by color
- usable click/touch targets
- text reflow without clipping
- responsive behavior
- reduced-motion preference for any animation

The interface should remain usable at approximately:

- 375 px
- 768 px
- 1024 px
- 1440 px

9. STREAMLIT CONSTRAINT

Work with Streamlit rather than fighting it.

Prefer:

- Streamlit components
- small centralized CSS overrides in app/styles.py
- existing application state/navigation patterns

Avoid brittle DOM selectors unless unavoidable.

Do not introduce JavaScript hacks or a separate frontend framework simply to
make rows clickable.

If Streamlit limitations prevent an exact traditional-email layout, implement
the simplest robust approximation while preserving accessibility and behavior.

10. SCOPE

Focus only on the email client and the sidebar elements needed to support it.

Do not redesign:

- Model Dashboard
- Dataset Dashboard
- Methodology page
- ML visualization pages

unless a tiny shared style change is required to prevent breakage.

Keep the code change as localized as possible.

OUTPUT

Implement the redesign in the existing repository.

After implementation, report:

1. files changed;
2. main UI decisions;
3. any Streamlit limitation that affected the design;
4. tests/commands run;
5. whether all tests passed.

VERIFICATION

Run the smallest relevant tests after the change, at minimum:

python -m pytest tests/test_mailbox_ui.py tests/test_streamlit_app.py tests/test_mail_service.py tests/test_mail_router.py tests/test_mail_repository.py -q

If those pass and the environment supports it, also run:

python -m pytest -m "not production_artifact" -q

Review the implementation against this checklist before finishing:

- no marketing hero on mailbox page
- search is visually primary
- email rows are compact and scannable
- unread/read hierarchy is obvious
- status badges are small
- normal email is visually calm
- phishing warning remains obvious
- email body precedes technical analysis
- technical details are progressively disclosed
- no gradients / glass / 3D / heavy shadows
- colors are restrained
- responsive text does not clip
- existing mailbox behavior remains intact
- ML/risk/security logic is unchanged

Do not stop after proposing a mockup. Make the code changes and verify them.
```

---

## 7. Cách dùng prompt

### Codex IDE / ChatGPT coding agent

Mở project VietMailGuard rồi dán toàn bộ prompt ở mục 6.

Nếu dùng Codex IDE, nên mở sẵn:

```text
app/app.py
app/pages/mail_client.py
app/styles.py
app/mailbox_ui.py
```

để model có context trực tiếp.

### Nếu vòng đầu vẫn còn quá nhiều chi tiết

Dùng follow-up ngắn:

```text
Reduce visual noise further.

Keep all existing behavior and layout structure, but:
- remove unnecessary borders
- remove promotional copy
- make mailbox rows denser
- reduce badge prominence
- make the email body more visually dominant
- keep only blue, neutral gray, amber, red, and subtle green where semantically necessary

Do not add new features.
```

### Nếu danh sách email vẫn giống dashboard

Dùng:

```text
Focus only on the mailbox list now.

Make it feel like a conventional inbox, not a security dashboard.

Sender, subject, preview, unread state, and timestamp should dominate.
Prediction/risk should become small secondary metadata.

Do not touch backend behavior or the detail page in this iteration.
```

---

## 8. Tiêu chí hoàn thành

Redesign được xem là đạt khi người dùng mở VietMailGuard và cảm nhận trước tiên:

> "Đây là một ứng dụng email đơn giản có thêm tính năng bảo mật."

Thay vì:

> "Đây là một dashboard Machine Learning/cybersecurity có chức năng đọc email."
