import os

class HTMLReportGenerator:
    """Generates a clean, executive dashboard showing evidence directly to instructors."""
    @staticmethod
    def write_report(report_data, families, output_path="collusion_report.html"):
        # --- FIX: Auto-handle if the user just passes a directory path ---
        if os.path.isdir(output_path) or output_path.endswith('/'):
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "collusion_report.html")
        else:
            # If a full file path like 'output/report.html' was passed, ensure parent dir exists
            parent_dir = os.path.dirname(output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Integrity Analytics - Solution Family Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
                h1, h2 {{ color: #1e293b; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #edf2f7; }}
                th {{ background-color: #0f172a; color: white; }}
                tr:hover {{ background-color: #f1f5f9; }}
                .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
                .badge-critical {{ background: #fee2e2; color: #991b1b; }}
                .badge-high {{ background: #ffedd5; color: #9a3412; }}
                .family-box {{ background: #edf2f7; padding: 10px; margin: 5px 0; border-left: 4px solid #3182ce; border-radius: 0 4px 4px 0; }}
            </style>
        </head>
        <body>
            <h1>Integrity Analytics Dashboard</h1>
            <p><strong>MSc Project Pipeline Verification Output</strong></p>
            <hr>
            
            <h2>Cohort Summary (Solution Families)</h2>
            <div class="metric-card">
                <p>Total Flagged Structural Families: {len([f for f, m in families.items() if len(m) > 1])}</p>
                {"".join([f"<div class='family-box'><strong>{name}</strong> ({len(m)} submissions): {', '.join(m)}</div>" for name, m in families.items() if len(m) > 1])}
            </div>

            <h2>High-Risk Anomalies (Family-Aware Ranking)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Assigned Family</th>
                        <th>Student A</th>
                        <th>Student B</th>
                        <th>Structural Similarity</th>
                        <th>Family Group Density</th>
                        <th>Risk Assessment</th>
                    </tr>
                </thead>
                <tbody>
        """
        for row in report_data:
            badge_class = "badge-critical" if row['risk_level'] == "CRITICAL" else "badge-high"
            html_content += f"""
                    <tr>
                        <td><code>{row['family']}</code></td>
                        <td>{row['student_a']}</td>
                        <td>{row['student_b']}</td>
                        <td><strong>{row['similarity']:.4f}</strong></td>
                        <td>{row['family_density']:.4f}</td>
                        <td><span class="badge {badge_class}">{row['risk_level']}</span></td>
                    </tr>
            """
            
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        
        # This will now write out to output/collusion_report.html cleanly
        with open(output_path, "w") as f:
            f.write(html_content)
