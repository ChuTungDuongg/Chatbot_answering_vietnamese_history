function LogoMark({ className = "" }) {
  return (
    <svg className={`logo-mark ${className}`} viewBox="0 0 64 64" role="img" aria-label="Logo Sử Việt AI">
      <rect className="logo-mark-bg" x="2" y="2" width="60" height="60" rx="14" />
      <path className="logo-mark-roof" d="M13 27.5 32 15l19 12.5-3.5 4L32 22 16.5 31.5 13 27.5Z" />
      <path className="logo-mark-lines" d="M19 32h26M22 38h20M25 44h14" />
      <path className="logo-mark-star" d="m47 12 1.6 3.2 3.6.5-2.6 2.5.6 3.6-3.2-1.7-3.2 1.7.6-3.6-2.6-2.5 3.6-.5L47 12Z" />
    </svg>
  );
}

export default LogoMark;
