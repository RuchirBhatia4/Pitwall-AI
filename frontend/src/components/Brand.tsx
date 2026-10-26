import Image from "next/image";

// Pitwall AI brand kit (public/branding): raster logos cut from the chosen
// wheel-gun logo sheet, so they are shown at their own aspect ratio, never
// redrawn. The colour logo is made for dark surfaces.

/** Header logo: the full logo on wide screens, the wheel gun on narrow ones. */
export function NavLogo() {
  return (
    <>
      <Image src="/branding/logos/pitwall-logo-navbar.png" alt="PitWall AI" width={600} height={94} priority className="hidden lg:block h-7 w-auto" />
      <Image src="/branding/logos/pitwall-logo-wheelgun-transparent.png" alt="PitWall AI" width={232} height={152} priority className="lg:hidden h-8 w-auto" />
    </>
  );
}

export function FooterLogo() {
  return <Image src="/branding/logos/pitwall-logo-footer-white.png" alt="PitWall AI" width={720} height={113} className="h-5 w-auto opacity-80" />;
}

/** Loading state: the strategy-wheel mark, pulsing unless the visitor prefers reduced motion. */
export function StrategyLoader({ label, className = "" }: { label: string; className?: string }) {
  return (
    <div className={`flex items-center gap-3 text-text-3 ${className}`} role="status">
      <Image src="/branding/icons/strategy-loader-512.png" alt="" width={512} height={512} className="pitwall-loader h-10 w-10" />
      {label}
    </div>
  );
}
