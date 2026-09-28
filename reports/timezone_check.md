# Kick-off timezone check

- E0 Saturday modal kick-offs: [('15:00', 357), ('17:30', 94), ('12:30', 70)] — expected 15:00 (UK)
- D1 Saturday modal kick-offs: [('14:30', 470), ('17:30', 96)] — expected 15:30 local / 14:30 UK
- SP1 Saturday modal kick-offs: [('20:00', 100), ('17:30', 95), ('15:15', 91)] — expected 21:00 local / 20:00 UK (evening)
- I1 Saturday modal kick-offs: [('19:45', 116), ('14:00', 115), ('17:00', 100)] — expected 20:45 local / 19:45 UK (evening)
- Real Madrid v Barcelona 26/10/2024 listed at 20:00 (actual 21:00 CEST = 20:00 BST)

**Conclusion:** times look like UK time (Europe/London).
Set `data.kickoff_timezone` and `kickoff_timezone_verified: true` in config.yaml accordingly.
