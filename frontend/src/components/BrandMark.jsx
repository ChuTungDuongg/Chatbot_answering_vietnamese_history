// One notched petal, repeated radially; keep the silhouette clear at avatar sizes.
const PETAL = "M32 28C28 25 22 19 23 12C23.4 8.5 26 6 29 6L32 9L35 6C38 6 40.6 8.5 41 12C42 19 36 25 32 28Z";

function BrandMark({ size = 24, className = "", label }) {
  return (
    <svg
      className={`brand-mark ${className}`.trim()}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 64 64"
      width={size}
      height={size}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
    >
      <g className="brand-mark-petals">
        {[0, 72, 144, 216, 288].map((angle) => (
          <path key={angle} d={PETAL} transform={`rotate(${angle} 32 32)`} />
        ))}
      </g>
      <circle className="brand-mark-center" cx="32" cy="32" r="4.5" />
    </svg>
  );
}

export default BrandMark;
