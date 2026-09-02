"""Dependency-free modern ttk theme; safe fallback when ttkbootstrap is absent."""

from __future__ import annotations


PALETTE = {
    "canvas": "#F4F7FB",
    "surface": "#FFFFFF",
    "text": "#172033",
    "muted": "#667085",
    "primary": "#2563EB",
    "primary_active": "#1D4ED8",
    "success": "#16803B",
    "border": "#E4E9F2",
}


def apply_modern_theme(root, ttk, tk_font) -> str:  # noqa: ANN001
    families = set(tk_font.families(root))
    family = "Segoe UI" if "Segoe UI" in families else "Microsoft YaHei UI"
    for name, size, weight in (
        ("TkDefaultFont", 10, "normal"),
        ("TkTextFont", 10, "normal"),
        ("TkHeadingFont", 10, "bold"),
    ):
        tk_font.nametofont(name).configure(family=family, size=size, weight=weight)
    root.configure(background=PALETTE["canvas"])
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("TFrame", background=PALETTE["canvas"])
    style.configure("Card.TFrame", background=PALETTE["surface"], relief="flat")
    style.configure("TLabel", background=PALETTE["canvas"], foreground=PALETTE["text"])
    style.configure("Card.TLabel", background=PALETTE["surface"], foreground=PALETTE["text"])
    style.configure("Title.TLabel", font=(family, 20, "bold"), foreground=PALETTE["text"])
    style.configure("PageTitle.TLabel", font=(family, 16, "bold"), foreground=PALETTE["text"])
    style.configure("CardTitle.TLabel", background=PALETTE["surface"], font=(family, 11, "bold"), foreground=PALETTE["text"])
    style.configure("Flow.TLabel", font=(family, 10, "bold"), foreground="#35536F")
    style.configure("Help.TLabel", foreground=PALETTE["muted"])
    style.configure("CardHelp.TLabel", background=PALETTE["surface"], foreground=PALETTE["muted"])
    style.configure("Status.TLabel", font=(family, 19, "bold"), foreground="#24547A")
    style.configure("Wake.Status.TLabel", font=(family, 19, "bold"), foreground=PALETTE["success"])
    style.configure("Stopped.Status.TLabel", font=(family, 19, "bold"), foreground="#666666")
    style.configure("CardValue.TLabel", font=(family, 11, "bold"))
    style.configure("CardValue.Card.TLabel", background=PALETTE["surface"], font=(family, 11, "bold"))
    style.configure("Primary.TButton", padding=(18, 9), background=PALETTE["primary"], foreground="#FFFFFF", borderwidth=0)
    style.map("Primary.TButton", background=[("active", PALETTE["primary_active"]), ("pressed", PALETTE["primary_active"])])
    style.configure("Secondary.TButton", padding=(14, 8), background="#EAF0FA", foreground="#344054", borderwidth=0)
    style.configure("TButton", padding=(12, 7))
    style.configure("TNotebook", background=PALETTE["canvas"], borderwidth=0)
    style.configure("TNotebook.Tab", padding=(18, 9), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", PALETTE["surface"]), ("!selected", "#E9EEF6")], foreground=[("selected", PALETTE["primary"])])
    style.configure("TLabelframe", background=PALETTE["surface"], borderwidth=0, relief="flat")
    style.configure("TLabelframe.Label", background=PALETTE["surface"], foreground=PALETTE["text"], font=(family, 11, "bold"))
    return family
