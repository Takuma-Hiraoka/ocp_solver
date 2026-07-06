#pragma once

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace ocp_solver {

class ScopeProfilerEntry {
 public:
  explicit ScopeProfilerEntry(const char* name) : name_(name) {
    if (isEnabled()) {
      registerEntry(this);
    }
  }

  void add(long long nanoseconds) {
    totalNanoseconds_.fetch_add(nanoseconds, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
  }

  const char* name() const { return name_; }
  long long totalNanoseconds() const { return totalNanoseconds_.load(std::memory_order_relaxed); }
  long long count() const { return count_.load(std::memory_order_relaxed); }

  static bool isEnabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("OCP_PROFILE_SCOPES");
      return value != nullptr && std::string(value) != "0";
    }();
    return enabled;
  }

 private:
  static void registerEntry(ScopeProfilerEntry* entry) {
    std::lock_guard<std::mutex> lock(mutex());
    entries().push_back(entry);
    static bool registeredAtExit = []() {
      std::atexit(&ScopeProfilerEntry::printSummary);
      return true;
    }();
    (void)registeredAtExit;
  }

  static void printSummary() {
    std::vector<ScopeProfilerEntry*> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex());
      snapshot = entries();
    }
    std::sort(snapshot.begin(), snapshot.end(), [](const ScopeProfilerEntry* lhs, const ScopeProfilerEntry* rhs) {
      return lhs->totalNanoseconds() > rhs->totalNanoseconds();
    });

    std::cerr << "\n[OCP scope profiler]\n";
    std::cerr << "  total_ms    avg_us      calls  scope\n";
    for (const ScopeProfilerEntry* entry : snapshot) {
      const long long count = entry->count();
      if (count == 0) {
        continue;
      }
      const double totalMs = static_cast<double>(entry->totalNanoseconds()) * 1.0e-6;
      const double avgUs = static_cast<double>(entry->totalNanoseconds()) / static_cast<double>(count) * 1.0e-3;
      std::cerr << std::setw(10) << std::fixed << std::setprecision(3) << totalMs
                << " " << std::setw(9) << std::setprecision(3) << avgUs
                << " " << std::setw(10) << count
                << "  " << entry->name() << "\n";
    }
  }

  static std::vector<ScopeProfilerEntry*>& entries() {
    static std::vector<ScopeProfilerEntry*> entries;
    return entries;
  }

  static std::mutex& mutex() {
    static std::mutex mutex;
    return mutex;
  }

  const char* name_;
  std::atomic<long long> totalNanoseconds_{0};
  std::atomic<long long> count_{0};
};

class ScopeProfiler {
 public:
  explicit ScopeProfiler(ScopeProfilerEntry& entry)
      : entry_(entry), enabled_(ScopeProfilerEntry::isEnabled()) {
    if (enabled_) {
      start_ = Clock::now();
    }
  }

  ~ScopeProfiler() {
    if (!enabled_) {
      return;
    }
    const auto end = Clock::now();
    entry_.add(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count());
  }

 private:
  using Clock = std::chrono::steady_clock;
  ScopeProfilerEntry& entry_;
  bool enabled_ = false;
  Clock::time_point start_;
};

}  // namespace ocp_solver

#define OCP_SOLVER_PROFILE_SCOPE(name)      \
  static ::ocp_solver::ScopeProfilerEntry   \
      ocp_solver_scope_profiler_entry(name); \
  ::ocp_solver::ScopeProfiler               \
      ocp_solver_scope_profiler(ocp_solver_scope_profiler_entry)
