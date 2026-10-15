// sandbox_run: tiny launcher used by eval/local_eval.py to run one untrusted program.
//
//   sandbox_run [--netns] [--cpu S] [--as B] [--data B] [--stack B] [--fsize B] [--nofile N]
//               [--wall SECONDS] --report-fd FD -- PROGRAM [ARGS...]
//
// * --netns: unshare(CLONE_NEWUSER|CLONE_NEWNET) (own uid/gid mapped to themselves) => no network.
// * --rofs (needs --netns): also a private mount namespace in which the current working directory is
//   bind-mounted read-write and every other mount (except /proc, /sys, /dev) is remounted read-only
//   (best effort; the number of mounts that could not be remounted is reported as rofs_failures).
// * forks; the child sets the rlimits (soft = hard), RLIMIT_CORE = 0, and execs PROGRAM with the
//   inherited stdin/stdout/stderr, cwd and environment.
// * the parent kills the child after --wall seconds (SIGKILL) and writes one JSON line with the
//   child's own rusage (wait4) to FD. Measuring the forked child (not this process) keeps ru_maxrss
//   free of the Python parent's RSS, which a vfork/exec chain would otherwise inherit.
// Exit status: 0 = report written; 125 = sandbox setup failure (message on stderr).
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sched.h>
#include <sys/mount.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_child = 0;
static volatile sig_atomic_t g_wall_killed = 0;

static void on_alarm(int sig) {
  (void)sig;
  if (g_child > 0) {
    g_wall_killed = 1;
    kill(g_child, SIGKILL);
  }
}

static int write_str(const char *path, const char *s) {
  int fd = open(path, O_WRONLY | O_CLOEXEC);
  if (fd < 0) return -1;
  size_t n = strlen(s);
  ssize_t w = write(fd, s, n);
  close(fd);
  return (w == (ssize_t)n) ? 0 : -1;
}

// Unescape the octal escapes (\040 etc.) used in /proc/self/mountinfo.
static void unescape(char *s) {
  char *o = s;
  for (char *p = s; *p; p++) {
    if (p[0] == '\\' && p[1] >= '0' && p[1] <= '7' && p[2] >= '0' && p[2] <= '7' && p[3] >= '0' && p[3] <= '7') {
      *o++ = (char)((p[1] - '0') * 64 + (p[2] - '0') * 8 + (p[3] - '0'));
      p += 3;
    } else {
      *o++ = *p;
    }
  }
  *o = 0;
}

static int has_opt(const char *opts, const char *name) {
  size_t n = strlen(name);
  for (const char *p = opts; (p = strstr(p, name)) != NULL; p += n) {
    if ((p == opts || p[-1] == ',') && (p[n] == ',' || p[n] == 0)) return 1;
  }
  return 0;
}

static int skip_mp(const char *mp) {
  const char *pre[] = {"/proc", "/sys", "/dev/pts", "/dev/mqueue", "/dev/hugepages", NULL};
  for (int i = 0; pre[i]; i++) {
    size_t n = strlen(pre[i]);
    if (!strncmp(mp, pre[i], n) && (mp[n] == 0 || mp[n] == '/')) return 1;
  }
  return !strcmp(mp, "/dev");  // keep /dev itself writable (device nodes); /dev/shm becomes read-only
}

// Returns number of mounts that could not be made read-only, or -1 on setup failure.
static int make_rofs(void) {
  char cwd[PATH_MAX];
  if (!getcwd(cwd, sizeof cwd)) return -1;
  if (unshare(CLONE_NEWNS) != 0) return -1;
  if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0) return -1;
  if (mount(cwd, cwd, NULL, MS_BIND | MS_REC, NULL) != 0) return -1;
  if (chdir(cwd) != 0) return -1;  // enter the new (rw) bind mount
  FILE *f = fopen("/proc/self/mountinfo", "r");
  if (!f) return -1;
  char line[8192];
  int failures = 0;
  while (fgets(line, sizeof line, f)) {
    char mp[4096], opts[1024];
    // id parent maj:min root mountpoint options ...
    if (sscanf(line, "%*d %*d %*s %*s %4095s %1023s", mp, opts) != 2) continue;
    unescape(mp);
    if (!strcmp(mp, cwd) || skip_mp(mp)) continue;
    if (has_opt(opts, "ro")) continue;
    unsigned long fl = MS_REMOUNT | MS_BIND | MS_RDONLY;
    if (has_opt(opts, "nosuid")) fl |= MS_NOSUID;
    if (has_opt(opts, "nodev")) fl |= MS_NODEV;
    if (has_opt(opts, "noexec")) fl |= MS_NOEXEC;
    if (has_opt(opts, "noatime")) fl |= MS_NOATIME;
    if (has_opt(opts, "nodiratime")) fl |= MS_NODIRATIME;
    if (has_opt(opts, "relatime")) fl |= MS_RELATIME;
    if (mount(NULL, mp, NULL, fl, NULL) != 0) {
      failures++;
      if (getenv("SANDBOX_RUN_DEBUG")) fprintf(stderr, "sandbox_run: rofs: cannot remount %s (%s): %s\n", mp, opts, strerror(errno));
    }
  }
  fclose(f);
  return failures;
}

static int set_rl(int res, long long v) {
  if (v < 0) return 0;
  struct rlimit rl;
  rl.rlim_cur = rl.rlim_max = (rlim_t)v;
  return setrlimit(res, &rl);
}

int main(int argc, char **argv) {
  int netns = 0, rofs = 0, rofs_failures = -1, report_fd = -1;
  long long cpu = -1, as = -1, data = -1, stack = -1, fsize = -1, nofile = -1;
  double wall = 0.0;
  int i = 1;
  for (; i < argc; i++) {
    const char *a = argv[i];
    if (!strcmp(a, "--")) { i++; break; }
    if (!strcmp(a, "--netns")) { netns = 1; continue; }
    if (!strcmp(a, "--rofs")) { rofs = 1; continue; }
    if (i + 1 >= argc) { fprintf(stderr, "sandbox_run: missing value for %s\n", a); return 125; }
    const char *v = argv[++i];
    if (!strcmp(a, "--cpu")) cpu = atoll(v);
    else if (!strcmp(a, "--as")) as = atoll(v);
    else if (!strcmp(a, "--data")) data = atoll(v);
    else if (!strcmp(a, "--stack")) stack = atoll(v);
    else if (!strcmp(a, "--fsize")) fsize = atoll(v);
    else if (!strcmp(a, "--nofile")) nofile = atoll(v);
    else if (!strcmp(a, "--wall")) wall = atof(v);
    else if (!strcmp(a, "--report-fd")) report_fd = atoi(v);
    else { fprintf(stderr, "sandbox_run: unknown option %s\n", a); return 125; }
  }
  if (i >= argc || report_fd < 0) {
    fprintf(stderr, "usage: sandbox_run [opts] --report-fd FD -- PROGRAM [ARGS...]\n");
    return 125;
  }
  if (fcntl(report_fd, F_SETFD, FD_CLOEXEC) != 0) { perror("sandbox_run: report fd"); return 125; }

  if (netns) {
    uid_t uid = getuid();
    gid_t gid = getgid();
    if (unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0) { perror("sandbox_run: unshare"); return 125; }
    char buf[64];
    write_str("/proc/self/setgroups", "deny");
    snprintf(buf, sizeof buf, "%u %u 1", (unsigned)uid, (unsigned)uid);
    if (write_str("/proc/self/uid_map", buf) != 0) { perror("sandbox_run: uid_map"); return 125; }
    snprintf(buf, sizeof buf, "%u %u 1", (unsigned)gid, (unsigned)gid);
    if (write_str("/proc/self/gid_map", buf) != 0) { perror("sandbox_run: gid_map"); return 125; }
    if (rofs) {
      rofs_failures = make_rofs();
      if (rofs_failures < 0) { perror("sandbox_run: rofs"); return 125; }
    }
  }

  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_alarm;
  sigaction(SIGALRM, &sa, NULL);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  pid_t pid = fork();
  if (pid < 0) { perror("sandbox_run: fork"); return 125; }
  if (pid == 0) {
    signal(SIGALRM, SIG_DFL);
    if (set_rl(RLIMIT_CPU, cpu) || set_rl(RLIMIT_AS, as) || set_rl(RLIMIT_DATA, data) ||
        set_rl(RLIMIT_STACK, stack) || set_rl(RLIMIT_FSIZE, fsize) || set_rl(RLIMIT_NOFILE, nofile) ||
        set_rl(RLIMIT_CORE, 0)) {
      perror("sandbox_run: setrlimit");
      _exit(125);
    }
    execv(argv[i], &argv[i]);
    perror("sandbox_run: execv");
    _exit(127);
  }
  g_child = pid;
  if (wall > 0) {
    struct itimerval it;
    memset(&it, 0, sizeof it);
    it.it_value.tv_sec = (time_t)wall;
    it.it_value.tv_usec = (suseconds_t)((wall - (double)(time_t)wall) * 1e6);
    if (it.it_value.tv_sec == 0 && it.it_value.tv_usec == 0) it.it_value.tv_usec = 1;
    setitimer(ITIMER_REAL, &it, NULL);
  }
  int st = 0;
  struct rusage ru;
  memset(&ru, 0, sizeof ru);
  pid_t w;
  do { w = wait4(pid, &st, 0, &ru); } while (w < 0 && errno == EINTR);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  struct itimerval z;
  memset(&z, 0, sizeof z);
  setitimer(ITIMER_REAL, &z, NULL);
  if (w < 0) { perror("sandbox_run: wait4"); return 125; }

  double cpu_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6;
  double wall_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  char out[512];
  int n = snprintf(out, sizeof out,
                   "{\"exit_code\": %d, \"signal\": %d, \"cpu_s\": %.6f, \"wall_s\": %.6f, "
                   "\"maxrss_kb\": %ld, \"wall_killed\": %d, \"rofs_failures\": %d}\n",
                   WIFEXITED(st) ? WEXITSTATUS(st) : -1, WIFSIGNALED(st) ? WTERMSIG(st) : 0, cpu_s, wall_s,
                   (long)ru.ru_maxrss, (int)g_wall_killed, rofs_failures);
  if (write(report_fd, out, (size_t)n) != n) return 125;
  close(report_fd);
  return 0;
}
