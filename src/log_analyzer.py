"""
Log Analysis Utility for FundTuneLab Data Collection

This module provides utilities to analyze and summarize data collection logs.
"""

import re
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from config.settings import RAW_DATA_DIR


class DataCollectionLogAnalyzer:
    """Analyzer for data collection log files."""

    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize the log analyzer.

        Args:
            log_file (Optional[Path]): Path to log file. Defaults to standard location.
        """
        self.log_file = log_file or RAW_DATA_DIR / "data_collection.log"

    def parse_log_entry(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parse a single log entry.

        Args:
            line (str): Log line to parse

        Returns:
            Optional[Dict[str, str]]: Parsed log entry or None if invalid
        """
        # Pattern: timestamp - logger_name - level - message
        pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.*?) - (\w+) - (.*)"
        match = re.match(pattern, line.strip())

        if match:
            timestamp_str, logger_name, level, message = match.groups()
            return {
                "timestamp": timestamp_str,
                "logger": logger_name,
                "level": level,
                "message": message,
                "raw_line": line.strip(),
            }
        return None

    def analyze_logs(self, hours_back: Optional[int] = None) -> Dict[str, any]:
        """
        Analyze log file and return comprehensive statistics.

        Args:
            hours_back (Optional[int]): Only analyze logs from last N hours

        Returns:
            Dict[str, any]: Analysis results
        """
        if not self.log_file.exists():
            return {
                "error": f"Log file not found: {self.log_file}",
                "file_exists": False,
            }

        analysis = {
            "file_path": str(self.log_file),
            "file_exists": True,
            "total_lines": 0,
            "parsed_lines": 0,
            "log_levels": defaultdict(int),
            "symbols_processed": set(),
            "symbols_successful": set(),
            "symbols_failed": set(),
            "error_messages": [],
            "circuit_breaker_events": [],
            "download_statistics": {
                "total_attempts": 0,
                "successful_downloads": 0,
                "failed_downloads": 0,
                "retry_events": 0,
            },
            "time_range": {"earliest": None, "latest": None},
        }

        # Calculate time cutoff if specified
        time_cutoff = None
        if hours_back:
            from datetime import timedelta

            time_cutoff = datetime.now() - timedelta(hours=hours_back)

        with open(self.log_file, "r") as f:
            for line in f:
                analysis["total_lines"] += 1

                parsed = self.parse_log_entry(line)
                if not parsed:
                    continue

                analysis["parsed_lines"] += 1

                # Parse timestamp
                try:
                    log_time = datetime.strptime(
                        parsed["timestamp"], "%Y-%m-%d %H:%M:%S,%f"
                    )

                    # Skip if outside time window
                    if time_cutoff and log_time < time_cutoff:
                        continue

                    # Update time range
                    if (
                        not analysis["time_range"]["earliest"]
                        or log_time < analysis["time_range"]["earliest"]
                    ):
                        analysis["time_range"]["earliest"] = log_time
                    if (
                        not analysis["time_range"]["latest"]
                        or log_time > analysis["time_range"]["latest"]
                    ):
                        analysis["time_range"]["latest"] = log_time

                except ValueError:
                    continue

                # Count log levels
                analysis["log_levels"][parsed["level"]] += 1

                message = parsed["message"]

                # Extract symbol names
                symbol_match = re.search(r"\b([A-Z]{2,5})\b", message)
                if symbol_match:
                    symbol = symbol_match.group(1)
                    # Filter out common words that match the pattern
                    if symbol not in [
                        "INFO",
                        "ERROR",
                        "WARNING",
                        "DEBUG",
                        "HTTP",
                        "CSV",
                    ]:
                        analysis["symbols_processed"].add(symbol)

                # Track download attempts
                if "Downloading data for" in message and "attempt" in message:
                    analysis["download_statistics"]["total_attempts"] += 1

                # Track successful downloads
                if "Successfully downloaded" in message and "rows for" in message:
                    analysis["download_statistics"]["successful_downloads"] += 1
                    if symbol_match:
                        symbol = symbol_match.group(1)
                        if symbol not in [
                            "INFO",
                            "ERROR",
                            "WARNING",
                            "DEBUG",
                            "HTTP",
                            "CSV",
                        ]:
                            analysis["symbols_successful"].add(symbol)

                # Track failed downloads
                if "Failed to download" in message or "Unable to download" in message:
                    analysis["download_statistics"]["failed_downloads"] += 1
                    if symbol_match:
                        symbol = symbol_match.group(1)
                        if symbol not in [
                            "INFO",
                            "ERROR",
                            "WARNING",
                            "DEBUG",
                            "HTTP",
                            "CSV",
                        ]:
                            analysis["symbols_failed"].add(symbol)

                # Track retry events
                if "Retrying in" in message:
                    analysis["download_statistics"]["retry_events"] += 1

                # Collect error messages
                if parsed["level"] in ["ERROR", "WARNING"]:
                    analysis["error_messages"].append(
                        {
                            "timestamp": parsed["timestamp"],
                            "level": parsed["level"],
                            "message": message,
                        }
                    )

                # Track circuit breaker events
                if "Circuit breaker" in message:
                    analysis["circuit_breaker_events"].append(
                        {"timestamp": parsed["timestamp"], "message": message}
                    )

        # Convert sets to lists for JSON serialization
        analysis["symbols_processed"] = sorted(list(analysis["symbols_processed"]))
        analysis["symbols_successful"] = sorted(list(analysis["symbols_successful"]))
        analysis["symbols_failed"] = sorted(list(analysis["symbols_failed"]))

        # Calculate success rate
        total_symbols = len(analysis["symbols_processed"])
        successful_symbols = len(analysis["symbols_successful"])
        analysis["symbol_success_rate"] = (
            successful_symbols / total_symbols * 100 if total_symbols > 0 else 0
        )

        return analysis

    def generate_summary_report(self, hours_back: Optional[int] = None) -> str:
        """
        Generate a human-readable summary report.

        Args:
            hours_back (Optional[int]): Only analyze logs from last N hours

        Returns:
            str: Formatted summary report
        """
        analysis = self.analyze_logs(hours_back)

        if not analysis["file_exists"]:
            return f"❌ {analysis['error']}"

        time_window = f" (last {hours_back} hours)" if hours_back else ""

        report = [
            f"📊 Data Collection Log Analysis{time_window}",
            "=" * 50,
            f"📁 Log file: {analysis['file_path']}",
            f"📝 Total log entries: {analysis['parsed_lines']}/{analysis['total_lines']}",
        ]

        if analysis["time_range"]["earliest"] and analysis["time_range"]["latest"]:
            report.append(
                f"⏰ Time range: {analysis['time_range']['earliest']} to {analysis['time_range']['latest']}"
            )

        # Log levels summary
        report.append("\n📈 Log Levels:")
        for level, count in sorted(analysis["log_levels"].items()):
            emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🐛"}.get(
                level, "📌"
            )
            report.append(f"  {emoji} {level}: {count}")

        # Symbol processing summary
        report.extend(
            [
                "\n🎯 Symbol Processing:",
                f"  📊 Total symbols processed: {len(analysis['symbols_processed'])}",
                f"  ✅ Successful: {len(analysis['symbols_successful'])} ({analysis['symbol_success_rate']:.1f}%)",
                f"  ❌ Failed: {len(analysis['symbols_failed'])}",
            ]
        )

        if analysis["symbols_successful"]:
            report.append(
                f"  🏆 Successful symbols: {', '.join(analysis['symbols_successful'])}"
            )

        if analysis["symbols_failed"]:
            report.append(
                f"  💥 Failed symbols: {', '.join(analysis['symbols_failed'])}"
            )

        # Download statistics
        stats = analysis["download_statistics"]
        report.extend(
            [
                "\n📥 Download Statistics:",
                f"  🔄 Total attempts: {stats['total_attempts']}",
                f"  ✅ Successful downloads: {stats['successful_downloads']}",
                f"  ❌ Failed downloads: {stats['failed_downloads']}",
                f"  🔁 Retry events: {stats['retry_events']}",
            ]
        )

        # Circuit breaker events
        if analysis["circuit_breaker_events"]:
            report.append(
                f"\n⚡ Circuit Breaker Events: {len(analysis['circuit_breaker_events'])}"
            )
            for event in analysis["circuit_breaker_events"][-3:]:  # Show last 3
                report.append(f"  🔌 {event['timestamp']}: {event['message']}")

        # Recent errors
        if analysis["error_messages"]:
            report.append(
                f"\n🚨 Recent Errors/Warnings ({len(analysis['error_messages'])} total):"
            )
            for error in analysis["error_messages"][-5:]:  # Show last 5
                emoji = "❌" if error["level"] == "ERROR" else "⚠️"
                report.append(f"  {emoji} {error['timestamp']}: {error['message']}")

        return "\n".join(report)


def analyze_data_collection_logs(hours_back: Optional[int] = None) -> str:
    """
    Convenience function to analyze data collection logs.

    Args:
        hours_back (Optional[int]): Only analyze logs from last N hours

    Returns:
        str: Formatted summary report
    """
    analyzer = DataCollectionLogAnalyzer()
    return analyzer.generate_summary_report(hours_back)


if __name__ == "__main__":
    # Example usage
    print(analyze_data_collection_logs())
