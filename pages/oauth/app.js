(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const api = window.AstrBotPluginPage;
  let description;
  let state = {status: "loading"};
  let busy = false;
  let revision = 0;
  let timer;
  let pollRequest;
  let disposed = false;
  let usageSnapshot = null;
  let usageBusy = false;
  let usageOperation = "";
  let usageRevision = 0;
  let usageError = "";
  const labels = {loading: "检查中", unbound: "未连接", authorized: "已连接", pending: "等待认证", cancelled: "已取消", denied: "授权未通过", expired: "认证码已过期", error: "授权异常", reauth_required: "需要重新授权"};
  const descriptions = {loading: "正在检查授权状态…", unbound: "连接账号后即可使用 Grok。", authorized: "授权凭据已保存，可供本插件的模型使用。", pending: "请在官方页面输入认证码并完成登录。", cancelled: "本次授权已取消，可以重新开始。", denied: "官方服务未批准本次授权，可以重新尝试。", expired: "请重新开始授权，获取新的认证码。", error: "请检查网络连接后重试。", reauth_required: "账号授权已失效，请重新连接。"};
  const usageStatuses = new Set(["success", "unknown", "unparseable", "unbound", "forbidden", "identity_unavailable", "rate_limited", "unavailable", "authorization_changed", "reauth_required"]);
  const usageLabels = {success: "查询成功", unknown: "额度未知", unparseable: "数据无法识别", unbound: "未连接", forbidden: "无权查询", identity_unavailable: "账号信息不可用", rate_limited: "查询受限", unavailable: "暂时不可用", authorization_changed: "授权已变化", reauth_required: "需要重新授权"};
  const usageDescriptions = {success: "额度快照来自 xAI 账号。", unknown: "上游没有返回可确认的额度比例。", unparseable: "上游返回的数据暂时无法识别。", unbound: "连接账号后可以手动查询额度。", forbidden: "当前账号无权查询额度。", identity_unavailable: "当前授权缺少查询额度所需的账号信息。", rate_limited: "查询过于频繁，请稍后再试。", unavailable: "额度服务暂时不可用，请稍后再试。", authorization_changed: "查询期间账号授权发生变化，请重新查询。", reauth_required: "当前授权已失效，请重新连接账号。"};

  function notice(message = "", error = false) {
    $("notice").textContent = message;
    $("notice").hidden = !message;
    $("notice").dataset.error = String(error);
  }

  function render() {
    const pending = state.status === "pending";
    const authorized = state.status === "authorized";
    $("status-label").textContent = labels[state.status] || "需要检查";
    $("status-label").dataset.state = state.status;
    $("status-description").textContent = descriptions[state.status] || "请检查当前授权状态。";
    $("start").hidden = pending || authorized;
    $("start").disabled = busy || !description;
    $("start").textContent = busy ? "正在连接…" : "连接 Grok 账号";
    $("refresh").disabled = busy || !description;
    $("disconnect").hidden = !authorized;
    $("disconnect").disabled = busy;
    $("confirm-disconnect").disabled = busy;
    $("device-panel").hidden = !pending;
    $("connected-panel").hidden = !authorized;
    $("cancel").disabled = busy;
    $("check-complete").disabled = busy;
    $("device-code").value = pending ? state.user_code || "" : "";
    $("official-url").value = pending ? state.verification_uri || "" : "";
    $("updated-at").textContent = "最近检查：" + new Date().toLocaleTimeString();
    renderUsage();
    countdown();
  }

  function clearUsage() {
    usageRevision += 1;
    usageSnapshot = null;
    usageError = "";
    renderUsage();
  }

  function finitePercent(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100 ? value : null;
  }

  function safeText(value) {
    return typeof value === "string" && value.length <= 80 ? value : null;
  }

  function snapshotData(result) {
    const raw = result && result.status === "ok" && result.data && typeof result.data === "object" ? result.data : result;
    if (!raw || typeof raw !== "object" || !usageStatuses.has(raw.status)) return null;
    return {
      status: raw.status,
      scope: raw.scope === "account_shared" ? "account_shared" : "unknown",
      period_type: ["weekly", "monthly"].includes(raw.period_type) ? raw.period_type : "unknown",
      used_percent: finitePercent(raw.used_percent),
      remaining_percent: finitePercent(raw.remaining_percent),
      reset_at: safeText(raw.reset_at),
      reset_at_local: safeText(raw.reset_at_local),
      observed_at: safeText(raw.observed_at),
      cached: raw.cached === true,
      stale: raw.stale === true,
    };
  }

  function formatPercent(value) {
    if (value === null) return "未知";
    return new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 1}).format(value) + "%";
  }

  function beijingTime(value) {
    if (!value) return "未知";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "未知";
    const text = new Intl.DateTimeFormat("zh-CN", {
      timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    }).format(date);
    return text + "（北京时间）";
  }

  function renderUsage() {
    const authorized = state.status === "authorized";
    const snapshot = usageSnapshot;
    let status = authorized ? "尚未查询" : "等待连接";
    let descriptionText = authorized ? "点击查询额度后显示账号额度快照。" : usageDescriptions.unbound;
    if (usageBusy) {
      status = usageOperation === "refresh" ? "正在刷新" : "正在查询";
      descriptionText = "正在请求额度快照…";
    } else if (usageError) {
      status = "查询失败";
      descriptionText = usageError;
    } else if (snapshot) {
      const partial = snapshot.status === "unknown" && (snapshot.used_percent !== null || snapshot.remaining_percent !== null);
      status = partial ? "部分信息" : usageLabels[snapshot.status];
      descriptionText = snapshot.stale ? "当前查询失败，以下内容仅为历史快照，不能代表当前可用额度。" : partial ? "部分额度信息未返回，已显示可以确认的比例。" : usageDescriptions[snapshot.status];
    }
    $("usage-status").textContent = status;
    $("usage-status").dataset.state = snapshot?.stale ? "stale" : snapshot?.status || (authorized ? "idle" : "unbound");
    $("usage-description").textContent = descriptionText;
    $("usage-stale").hidden = !snapshot?.stale;
    const canShowPercent = snapshot && (["success", "unknown"].includes(snapshot.status) || snapshot.stale);
    $("usage-primary-label").textContent = snapshot?.stale ? "历史剩余额度快照" : "剩余额度";
    $("usage-primary").textContent = formatPercent(canShowPercent ? snapshot.remaining_percent : null);
    $("usage-used").textContent = formatPercent(canShowPercent ? snapshot.used_percent : null);
    $("usage-scope").textContent = snapshot?.scope === "account_shared" ? "账号共享" : "未确认";
    $("usage-period").textContent = snapshot?.period_type === "weekly" ? "每周" : snapshot?.period_type === "monthly" ? "每月" : "未知";
    $("usage-reset").textContent = beijingTime(snapshot?.reset_at_local || snapshot?.reset_at);
    $("usage-observed").textContent = beijingTime(snapshot?.observed_at);
    $("usage-cache").textContent = snapshot?.cached ? (snapshot.stale ? "历史缓存快照" : "缓存结果") : "";
    $("usage-cache").hidden = !snapshot?.cached;
    $("usage-query").disabled = usageBusy || busy || !authorized;
    $("usage-refresh").disabled = usageBusy || busy || !authorized;
    $("usage-query").textContent = usageBusy && usageOperation === "query" ? "正在查询…" : "查询额度";
    $("usage-refresh").textContent = usageBusy && usageOperation === "refresh" ? "正在刷新…" : "强制刷新";
  }

  async function queryUsage(forceRefresh) {
    if (usageBusy || busy || disposed || state.status !== "authorized") return;
    usageBusy = true;
    usageOperation = forceRefresh ? "refresh" : "query";
    usageSnapshot = null;
    usageError = "";
    const generation = usageRevision;
    renderUsage();
    try {
      const result = await api.apiGet("usage", forceRefresh ? {refresh: "true"} : undefined);
      if (generation !== usageRevision || disposed || state.status !== "authorized") return;
      usageSnapshot = snapshotData(result);
      if (!usageSnapshot) usageError = "额度服务返回了无法识别的状态。";
    } catch {
      if (generation === usageRevision && !disposed) usageError = "暂时无法查询额度，请稍后重试。";
    } finally {
      usageBusy = false;
      usageOperation = "";
      renderUsage();
    }
  }

  function countdown() {
    const seconds = Math.max(0, Math.ceil((state.expires_at || 0) - Date.now() / 1000));
    $("countdown").textContent = seconds ? `有效期 ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}` : "正在确认授权结果";
  }

  function schedule() {
    clearTimeout(timer);
    if (!disposed && !busy && state.status === "pending") timer = setTimeout(() => refresh(false), 3000);
  }

  async function refresh(manual) {
    if (busy || disposed) return;
    if (pollRequest) return pollRequest;
    const generation = revision;
    pollRequest = (async () => {
      try {
        const result = await api.apiGet("auth/status");
        if (generation !== revision || disposed) return;
        const previousStatus = state.status;
        state = result;
        if (state.status !== "authorized" || previousStatus !== "authorized") clearUsage();
        render();
        notice(manual && state.status === "pending" ? "仍在等待官方授权结果，页面会继续自动检查。" : "");
      } catch {
        if (generation === revision && !disposed) notice("暂时无法检查状态，请稍后重试。", true);
      } finally {
        pollRequest = null;
        schedule();
      }
    })();
    return pollRequest;
  }

  function officialUrl() {
    const url = new URL(state.verification_uri);
    if (url.protocol !== "https:" || !["accounts.x.ai", "auth.x.ai"].includes(url.hostname) || url.username || url.password || (url.port && url.port !== "443")) throw new Error("Unexpected authorization destination");
    return url.href;
  }

  function openOfficial() {
    try {
      const url = officialUrl();
      const target = window.open("about:blank", "_blank");
      if (target) { target.opener = null; target.location.href = url; }
      // Sandboxed native plugin pages may disallow popups. Keep the official
      // URL and device code usable without depending on popup support.
      if (!target) $("open-help").hidden = false;
    } catch {
      $("open-help").hidden = false;
    }
  }

  async function copyInput(id, caption) {
    const input = $(id);
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(input.value);
      } else {
        input.focus();
        input.select();
        if (!document.execCommand("copy")) throw new Error("Clipboard unavailable");
      }
      notice(caption + "已复制。");
    } catch {
      input.focus();
      input.select();
      notice(caption + "已选中，请手动复制。");
    }
  }

  async function action(endpoint, body) {
    if (busy) return;
    busy = true;
    revision += 1;
    clearUsage();
    clearTimeout(timer);
    notice();
    render();
    try {
      state = await api.apiPost(endpoint, body);
      $("disconnect-confirm").hidden = true;
      $("open-help").hidden = true;
      render();
    } catch {
      notice("操作未完成，请检查网络连接后重试。", true);
    } finally {
      busy = false;
      render();
      schedule();
    }
  }

  $("start").addEventListener("click", () => action("auth/start", {
    confirmed_client_id: description.client_id,
    client_profile: description.client_profile,
  }));
  $("refresh").addEventListener("click", () => refresh(true));
  $("check-complete").addEventListener("click", () => refresh(true));
  $("cancel").addEventListener("click", () => action("auth/cancel", {flow_id: state.flow_id}));
  $("disconnect").addEventListener("click", () => { $("disconnect-confirm").hidden = false; });
  $("keep-connected").addEventListener("click", () => { $("disconnect-confirm").hidden = true; });
  $("confirm-disconnect").addEventListener("click", () => action("auth/disconnect", {}));
  $("open-official").addEventListener("click", openOfficial);
  $("copy-code").addEventListener("click", () => copyInput("device-code", "认证码"));
  $("copy-url").addEventListener("click", () => copyInput("official-url", "官方链接"));
  $("usage-query").addEventListener("click", () => queryUsage(false));
  $("usage-refresh").addEventListener("click", () => queryUsage(true));
  const clock = setInterval(countdown, 1000);
  window.addEventListener("pagehide", () => {disposed = true; clearTimeout(timer); clearInterval(clock);});

  async function initialize() {
    if (!api || window.parent === window) { notice("请从 AstrBot 的插件页面打开授权设置。", true); return; }
    try {
      await api.ready();
      description = await api.apiGet("auth/client");
      $("client-id").textContent = description.client_id;
      await refresh(false);
      render();
    } catch {
      notice("授权设置加载失败，请刷新页面重试。", true);
    }
  }
  initialize();
})();
